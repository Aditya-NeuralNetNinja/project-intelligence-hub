"""Project record data model for structured extraction from PDF reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ProjectRecord:
    """Canonical structured fields parsed from a single PDF project report."""

    # Identification
    source: str
    project_id: Optional[str] = None
    project_name: Optional[str] = None

    # Classification
    industry_code: Optional[str] = None
    project_type: Optional[str] = None
    sector: Optional[str] = None
    sic_code: Optional[str] = None
    sic_product: Optional[str] = None

    # Financial
    tiv_amount: Optional[float] = None
    tiv_currency: Optional[str] = None

    # Status
    status: Optional[str] = None
    status_reason: Optional[str] = None
    project_probability: Optional[str] = None

    # Timeline
    last_update: Optional[str] = None
    initial_release: Optional[str] = None
    pec_timing: Optional[str] = None
    pec_activity: Optional[str] = None

    # Location
    address: Optional[str] = None
    city_state_line: Optional[str] = None
    zone_county: Optional[str] = None

    # Plant Info
    plant_owner: Optional[str] = None
    plant_parent: Optional[str] = None
    plant_name: Optional[str] = None
    plant_id: Optional[str] = None
    unit_name: Optional[str] = None

    # Contacts
    project_manager: Optional[str] = None
    project_manager_company: Optional[str] = None
    project_manager_title: Optional[str] = None
    project_manager_email: Optional[str] = None
    project_manager_phone: Optional[str] = None
    engineer_company: Optional[str] = None
    ec_firm: Optional[str] = None
    phone: Optional[str] = None

    # Technical
    scope_text: Optional[str] = None
    project_capacity: Optional[str] = None
    environmental: Optional[str] = None
    construction_labor: Optional[str] = None
    operations_labor: Optional[str] = None
    fuel_type: Optional[str] = None

    # Derived text sections
    schedule_text: Optional[str] = None
    details_text: Optional[str] = None

    @property
    def owner_company(self) -> Optional[str]:
        """Alias for plant_owner (backward compatibility)."""
        return self.plant_owner

    def get_unique_key(self) -> str:
        return self.project_id or self.project_name or self.source

    def has_budget_info(self) -> bool:
        return self.tiv_amount is not None and self.tiv_currency is not None

    def has_location_info(self) -> bool:
        return any([self.address, self.city_state_line, self.zone_county])

    def has_timeline_info(self) -> bool:
        return bool(self.schedule_text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary with non-None fields only."""
        return {
            k: v for k, v in {
                "source": self.source,
                "project_id": self.project_id,
                "project_name": self.project_name,
                "industry_code": self.industry_code,
                "project_type": self.project_type,
                "sector": self.sector,
                "sic_code": self.sic_code,
                "sic_product": self.sic_product,
                "tiv_amount": self.tiv_amount,
                "tiv_currency": self.tiv_currency,
                "status": self.status,
                "status_reason": self.status_reason,
                "project_probability": self.project_probability,
                "last_update": self.last_update,
                "initial_release": self.initial_release,
                "pec_timing": self.pec_timing,
                "pec_activity": self.pec_activity,
                "address": self.address,
                "city_state_line": self.city_state_line,
                "zone_county": self.zone_county,
                "plant_owner": self.plant_owner,
                "plant_parent": self.plant_parent,
                "plant_name": self.plant_name,
                "plant_id": self.plant_id,
                "unit_name": self.unit_name,
                "project_manager": self.project_manager,
                "project_manager_company": self.project_manager_company,
                "project_manager_title": self.project_manager_title,
                "project_manager_email": self.project_manager_email,
                "project_manager_phone": self.project_manager_phone,
                "engineer_company": self.engineer_company,
                "ec_firm": self.ec_firm,
                "phone": self.phone,
                "scope_text": self.scope_text,
                "project_capacity": self.project_capacity,
                "environmental": self.environmental,
                "construction_labor": self.construction_labor,
                "operations_labor": self.operations_labor,
                "fuel_type": self.fuel_type,
                "schedule_text": self.schedule_text,
                "details_text": self.details_text,
            }.items() if v is not None
        }


@dataclass
class Milestone:
    """A project milestone extracted from schedule text."""

    name: str
    date_text: str = ""
    sentence: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "dateText": self.date_text, "sentence": self.sentence}


@dataclass
class GeoComponents:
    """Parsed geographic components from city/state line."""

    city: Optional[str] = None
    state: Optional[str] = None
    postal: Optional[str] = None
    country: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {"city": self.city, "state": self.state, "postal": self.postal, "country": self.country}
