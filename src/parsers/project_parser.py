"""Project report parser for semi-structured PDF documents."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from src.models.project import GeoComponents, Milestone, ProjectRecord
from src.config import get_logger

logger = get_logger(__name__)


class ProjectReportParser:
    """Comprehensive parser for semi-structured project report PDFs."""

    # Identification patterns
    PATTERN_PROJECT_ID = r"Project ID:\s*([0-9]+)"
    PATTERN_PROJECT_NAME = r"Project Name\s+(.+?)\s+PEC Activity Diagram"

    # Classification patterns
    PATTERN_INDUSTRY_CODE = r"Industry Code\s+([0-9]+\s+[A-Za-z\s&\(\)]+?)(?:\s+Project Type)"
    PATTERN_PROJECT_TYPE = r"Project Type\s+([A-Za-z]+)"
    PATTERN_SECTOR = r"Sector\s+([A-Za-z\s]+?)(?:\s+SIC Product|\s+Status)"
    PATTERN_SIC_CODE = r"SIC Code\s+([0-9]+\s+[A-Za-z\s&,\[\]]+?)(?:\s+Sector)"
    PATTERN_SIC_PRODUCT = r"SIC Product\s+([0-9\*]+\s+[A-Za-z\s,\(\)\-]+?)(?:\s+Status)"

    # Financial patterns
    PATTERN_TIV_USD = r"TIV \(USD\)\s*([0-9,]+)"
    PATTERN_TIV_CNY = r"TIV \(CNY\)\s*([0-9,]+)"

    # Status patterns
    PATTERN_STATUS = r"Status\s+([A-Za-z]+)\s+Last Update"
    PATTERN_STATUS_REASON = r"Status Reason\s+(.+?)\s+Environmental"
    PATTERN_PROJECT_PROBABILITY = r"Project Probability\s+([A-Za-z]+\s*\([0-9\-]+%\))"

    # Timeline patterns
    PATTERN_LAST_UPDATE = r"Last Update\s+([0-9]{2}-[A-Za-z]{3}-[0-9]{4})"
    PATTERN_INITIAL_RELEASE = r"Initial Release\s+([0-9]{2}-[A-Za-z]{3}-[0-9]{4})"
    PATTERN_PEC_TIMING = r"PEC.\s*Timing\s+([A-Z][0-9])"
    PATTERN_PEC_ACTIVITY = r"PEC.\s*Activity\s+([A-Za-z\s\-]+?)(?:\s+Project Probability)"

    # Location patterns
    PATTERN_LOCATION = r"Location\s+(.+?)\s+Phone"
    PATTERN_CITY_STATE = r"City/State\s+(.+?)\s+Zone/County"
    PATTERN_ZONE_COUNTY = r"Zone/County\s+(.+?)\s+Project Responsibility"
    PATTERN_PHONE = r"Phone\s+(\+?[0-9\s\-]+)"

    # Plant info patterns
    PATTERN_PLANT_OWNER = r"Plant Owner\s+([A-Za-z\s&,\.]+?)(?:\s+Plant Parent)"
    PATTERN_PLANT_PARENT = r"Plant Parent\s+([A-Za-z\s&,\.]+?)(?:\s+Plant Name|\s+Unit Name)"
    PATTERN_PLANT_NAME = r"Plant Name\s+([A-Za-z\s&,\.]+?)(?:\s+Unit Name|\s+Plant ID)"
    PATTERN_PLANT_ID = r"Plant ID\s+([0-9]+)"
    PATTERN_UNIT_NAME = r"Unit Name\s+([A-Za-z0-9\s&]+?)(?:\s+Plant ID|\s+Location)"

    # Contact patterns
    PATTERN_PROJECT_MANAGER = r"Project Manager\s+([A-Za-z\s&,\.]+?)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:\d|No\.|[A-Z][a-z]+\s+(?:Road|Street|Drive|Ave|Suite|Manager))"
    PATTERN_ENGINEER = r"Eng\s+([A-Za-z\s&,\.]+?)\s+(?:[A-Z][a-z]+\s+[A-Z][a-z]+|[0-9])"
    PATTERN_EC_FIRM = r"E&C\s+([A-Za-z\s&,\.]+?)\s+(?:[A-Z][a-z]+\s+[A-Z][a-z]+|[0-9])"
    PATTERN_EMAIL = r"\[E-Mail\]\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"

    # Technical patterns
    PATTERN_SCOPE = r"Scope\s+(.+?)\s+Schedule\s+"
    PATTERN_PROJECT_CAPACITY = r"Project Capacity\s+(?:Planned\s+)?([0-9,]+\s*(?:MW|BBL|Megawatts)[^\n]*)"
    PATTERN_ENVIRONMENTAL = r"Environmental\s+(Air\s*\([A-Z]\)[^C]*?)(?:\s+Construction Labor)"
    PATTERN_CONSTRUCTION_LABOR = r"Construction Labor Preference\s+([A-Za-z\-]+)"
    PATTERN_OPERATIONS_LABOR = r"Operations Labor Preference\s+([A-Za-z\-]+)"
    PATTERN_FUEL_TYPE = r"Project Fuel Type\s+([A-Za-z]+)"

    # Schedule/details patterns
    PATTERN_SCHEDULE = r"Schedule\s+(.+?)\bDetails\b"
    PATTERN_SCHEDULE_FALLBACK = r"Schedule\s+(.+?)\s+Engineering\s+(?:Civil|Contracting|Electrical)"
    PATTERN_DETAILS = r"Details\s+(.+?)\s+Engineering\s+(?:Civil|Contracting)"

    # Milestone pattern
    PATTERN_MILESTONE = (
        r"(?P<name>[A-Za-z0-9\-\s&/]+?)\s+"
        r"(?P<date>(?:[1-4]Q\d{2,4}|\d{4}|[A-Za-z]{3}-\d{4})(?:\s*\([^\)]*\))?)"
    )

    CHALLENGE_KEYWORDS = r"funding|partners|agreement|RFQ|bid|cancelled|delay|escalat"
    PATTERN_GEO = r"^(?P<city>[^,]+),\s*(?P<state>[^\d]+?)\s+(?P<postal>\d+)\s+(?P<country>.+)$"

    def __init__(self) -> None:
        self._compiled_patterns: Dict[str, re.Pattern] = {}

    def _get_pattern(self, pattern: str, flags: int = 0) -> re.Pattern:
        key = f"{pattern}:{flags}"
        if key not in self._compiled_patterns:
            self._compiled_patterns[key] = re.compile(pattern, flags)
        return self._compiled_patterns[key]

    def _find_match(self, text: str, pattern: str, flags: int = 0) -> Optional[str]:
        compiled = self._get_pattern(pattern, flags)
        match = compiled.search(text)
        return match.group(1).strip() if match else None

    def _find_all_matches(self, text: str, pattern: str, flags: int = 0) -> List[str]:
        compiled = self._get_pattern(pattern, flags)
        return [m.group(1).strip() for m in compiled.finditer(text)]

    @staticmethod
    def _money_to_float(value: str) -> Optional[float]:
        try:
            return float(value.replace(",", ""))
        except (ValueError, AttributeError):
            return None

    def _extract_project_manager(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract project manager name, company, and email."""
        pm_pattern = self._get_pattern(self.PATTERN_PROJECT_MANAGER, re.IGNORECASE)
        pm_match = pm_pattern.search(text)

        name, company, email = None, None, None

        if pm_match:
            company = pm_match.group(1).strip()
            name = pm_match.group(2).strip()
            pm_section = text[pm_match.start():pm_match.start() + 500]
            email_match = re.search(self.PATTERN_EMAIL, pm_section)
            if email_match:
                email = email_match.group(1)
            logger.info(f"Found Project Manager: {name} ({company})")

        return name, company, email

    def parse(self, text: str, source_name: str) -> ProjectRecord:
        """Parse a report into a ProjectRecord with comprehensive field extraction."""
        normalized = re.sub(r"\s+", " ", text)

        # Identification
        project_id = self._find_match(normalized, self.PATTERN_PROJECT_ID)
        project_name = self._find_match(normalized, self.PATTERN_PROJECT_NAME, re.IGNORECASE)

        # Classification
        industry_code = self._find_match(normalized, self.PATTERN_INDUSTRY_CODE, re.IGNORECASE)
        project_type = self._find_match(normalized, self.PATTERN_PROJECT_TYPE, re.IGNORECASE)
        sector = self._find_match(normalized, self.PATTERN_SECTOR, re.IGNORECASE)
        sic_code = self._find_match(normalized, self.PATTERN_SIC_CODE, re.IGNORECASE)
        sic_product = self._find_match(normalized, self.PATTERN_SIC_PRODUCT, re.IGNORECASE)

        # Financial
        tiv_usd = self._find_match(normalized, self.PATTERN_TIV_USD)
        tiv_cny = self._find_match(normalized, self.PATTERN_TIV_CNY)
        tiv_amount: Optional[float] = None
        tiv_currency: Optional[str] = None
        if tiv_usd:
            tiv_amount = self._money_to_float(tiv_usd)
            tiv_currency = "USD"
        elif tiv_cny:
            tiv_amount = self._money_to_float(tiv_cny)
            tiv_currency = "CNY"

        # Status
        status = self._find_match(normalized, self.PATTERN_STATUS, re.IGNORECASE)
        status_reason = self._find_match(normalized, self.PATTERN_STATUS_REASON, re.IGNORECASE)
        project_probability = self._find_match(normalized, self.PATTERN_PROJECT_PROBABILITY, re.IGNORECASE)

        # Timeline
        last_update = self._find_match(normalized, self.PATTERN_LAST_UPDATE)
        initial_release = self._find_match(normalized, self.PATTERN_INITIAL_RELEASE)
        pec_timing = self._find_match(normalized, self.PATTERN_PEC_TIMING, re.IGNORECASE)
        pec_activity = self._find_match(normalized, self.PATTERN_PEC_ACTIVITY, re.IGNORECASE)

        # Location
        address = self._find_match(normalized, self.PATTERN_LOCATION, re.IGNORECASE)
        city_state_line = self._find_match(normalized, self.PATTERN_CITY_STATE, re.IGNORECASE)
        zone_county = self._find_match(normalized, self.PATTERN_ZONE_COUNTY, re.IGNORECASE)
        phone = self._find_match(normalized, self.PATTERN_PHONE)

        # Plant info
        plant_owner = self._find_match(normalized, self.PATTERN_PLANT_OWNER, re.IGNORECASE)
        plant_parent = self._find_match(normalized, self.PATTERN_PLANT_PARENT, re.IGNORECASE)
        plant_name = self._find_match(normalized, self.PATTERN_PLANT_NAME, re.IGNORECASE)
        plant_id = self._find_match(normalized, self.PATTERN_PLANT_ID)
        unit_name = self._find_match(normalized, self.PATTERN_UNIT_NAME, re.IGNORECASE)

        # Contacts
        project_manager, project_manager_company, project_manager_email = self._extract_project_manager(normalized)
        engineer_company = self._find_match(normalized, self.PATTERN_ENGINEER, re.IGNORECASE)
        ec_firm = self._find_match(normalized, self.PATTERN_EC_FIRM, re.IGNORECASE)

        # Technical
        scope_text = self._find_match(normalized, self.PATTERN_SCOPE, re.IGNORECASE | re.DOTALL)
        project_capacity = self._find_match(normalized, self.PATTERN_PROJECT_CAPACITY, re.IGNORECASE)
        environmental = self._find_match(normalized, self.PATTERN_ENVIRONMENTAL, re.IGNORECASE)
        construction_labor = self._find_match(normalized, self.PATTERN_CONSTRUCTION_LABOR, re.IGNORECASE)
        operations_labor = self._find_match(normalized, self.PATTERN_OPERATIONS_LABOR, re.IGNORECASE)
        fuel_type = self._find_match(normalized, self.PATTERN_FUEL_TYPE, re.IGNORECASE)

        # Schedule/details
        schedule_text = self._find_match(normalized, self.PATTERN_SCHEDULE, re.IGNORECASE | re.DOTALL)
        if not schedule_text:
            schedule_text = self._find_match(normalized, self.PATTERN_SCHEDULE_FALLBACK, re.IGNORECASE | re.DOTALL)
        details_text = self._find_match(normalized, self.PATTERN_DETAILS, re.IGNORECASE | re.DOTALL)

        extracted_count = sum(1 for v in [
            project_id, project_name, industry_code, project_type, sector,
            tiv_amount, status, plant_owner, project_manager, scope_text,
            schedule_text, pec_timing, pec_activity
        ] if v is not None)
        logger.info(f"Extracted {extracted_count}/13 key fields from {source_name}")

        return ProjectRecord(
            source=source_name,
            project_id=project_id,
            project_name=project_name,
            industry_code=industry_code,
            project_type=project_type,
            sector=sector,
            sic_code=sic_code,
            sic_product=sic_product,
            tiv_amount=tiv_amount,
            tiv_currency=tiv_currency,
            status=status,
            status_reason=status_reason,
            project_probability=project_probability,
            last_update=last_update,
            initial_release=initial_release,
            pec_timing=pec_timing,
            pec_activity=pec_activity,
            address=address,
            city_state_line=city_state_line,
            zone_county=zone_county,
            plant_owner=plant_owner,
            plant_parent=plant_parent,
            plant_name=plant_name,
            plant_id=plant_id,
            unit_name=unit_name,
            project_manager=project_manager,
            project_manager_company=project_manager_company,
            project_manager_email=project_manager_email,
            engineer_company=engineer_company,
            ec_firm=ec_firm,
            phone=phone,
            scope_text=scope_text,
            project_capacity=project_capacity,
            environmental=environmental,
            construction_labor=construction_labor,
            operations_labor=operations_labor,
            fuel_type=fuel_type,
            schedule_text=schedule_text,
            details_text=details_text,
        )

    def extract_milestones(self, schedule_text: Optional[str]) -> List[Milestone]:
        """Extract milestone-like statements from schedule text."""
        if not schedule_text:
            return []

        milestones: List[Milestone] = []
        pattern = self._get_pattern(self.PATTERN_MILESTONE)

        for match in pattern.finditer(schedule_text):
            name = match.group("name").strip()
            date_text = match.group("date").strip()
            if len(name) >= 3 and name.lower() not in ("the", "and", "for", "with"):
                milestones.append(Milestone(
                    name=name,
                    date_text=date_text,
                    sentence=schedule_text[max(0, match.start()-50):match.end()+20].strip(),
                ))

        if not milestones and schedule_text.strip():
            milestones.append(Milestone(name="Schedule", date_text="", sentence=schedule_text.strip()[:200]))

        return milestones

    def derive_challenges(self, record: ProjectRecord) -> List[str]:
        """Derive candidate challenges/constraints from record fields."""
        candidates: List[str] = []

        if record.status_reason:
            candidates.append(f"Status reason: {record.status_reason}")
        if record.details_text:
            candidates.append(record.details_text)
        if record.schedule_text and re.search(self.CHALLENGE_KEYWORDS, record.schedule_text, re.IGNORECASE):
            candidates.append("Dependencies / commercial gating mentioned in schedule (funding, partners, RFQs/bids).")
        if record.status and record.status.lower() == "cancelled":
            candidates.append("Project status is Cancelled.")

        seen: set = set()
        cleaned: List[str] = []
        for candidate in candidates:
            candidate = candidate.strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                cleaned.append(candidate)
        return cleaned

    def parse_city_state_country(self, city_state_line: Optional[str]) -> GeoComponents:
        """Parse City/State line into structured components."""
        if not city_state_line:
            return GeoComponents()

        line = city_state_line.strip()
        pattern = self._get_pattern(self.PATTERN_GEO)
        match = pattern.match(line)

        if not match:
            return GeoComponents(city=line)

        return GeoComponents(
            city=match.group("city").strip(),
            state=match.group("state").strip(),
            postal=match.group("postal").strip(),
            country=match.group("country").strip(),
        )


_default_parser: Optional[ProjectReportParser] = None


def get_parser() -> ProjectReportParser:
    """Get the default parser instance (singleton)."""
    global _default_parser
    if _default_parser is None:
        _default_parser = ProjectReportParser()
    return _default_parser
