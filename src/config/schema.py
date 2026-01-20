"""Schema policy for LLM-driven graph extraction."""

from __future__ import annotations

from typing import List


class SchemaPolicy:
    """Defines allowed node labels and relationship types for LLM graph extraction.

    The LLMGraphTransformer benefits from explicit schema constraints. This schema
    is intentionally broad to support diverse project report questions (stakeholders,
    contracts, permitting, schedule, finance, risks, etc.).
    """

    ALLOWED_NODES: List[str] = [
        # Document structure
        "Project", "Report", "Document", "Section", "Chunk", "Source", "Evidence",
        # Organizations
        "Organization", "Company", "Owner", "ParentCompany", "Client", "Customer",
        "Partner", "JV", "Consortium", "Contractor", "Subcontractor", "Vendor", "Supplier",
        "Consultant", "EngineeringFirm", "EPC", "EPCM", "Operator",
        "GovernmentAgency", "Regulator", "Stakeholder",
        # People
        "Person", "Role", "Team", "Department",
        # Geography
        "Location", "Address", "City", "State", "Province", "Region", "Country", "County",
        "Zone", "Port", "Site", "Plant",
        # Finance
        "Budget", "Cost", "Capex", "Opex", "Estimate", "Investment", "Funding",
        "Currency", "TIV", "Revenue", "Tariff", "Price",
        # Timeline
        "Timeline", "Schedule", "Milestone", "Phase", "Stage", "Date", "Quarter", "Year",
        "Duration", "StartDate", "EndDate",
        # Technical
        "Industry", "Sector", "Market", "Demand", "Product", "Output", "Capacity",
        "Feedstock", "Fuel", "Technology", "Process", "Equipment", "Unit", "System", "Utility",
        "Specification", "Standard",
        # Contracts
        "Contract", "Agreement", "Tender", "Bid", "RFQ", "Procurement", "Permit",
        "WorkPackage", "Deliverable", "Requirement", "KPI", "Metric",
        # Status
        "Status", "StatusReason", "Decision", "Change", "Assumption", "Dependency",
        "Risk", "Issue", "Challenge", "Constraint", "Delay", "Cancellation",
        # ESG
        "EnvironmentalAspect", "Emissions", "Wastewater", "Water", "Waste", "Safety",
        "Regulation", "Compliance",
    ]

    ALLOWED_RELATIONSHIPS: List[str] = [
        # Document structure
        "HAS_REPORT", "HAS_DOCUMENT", "HAS_SECTION", "HAS_CHUNK", "HAS_EVIDENCE",
        "EVIDENCED_BY", "SUPPORTED_BY", "MENTIONS", "ABOUT",
        # Lifecycle
        "HAS_STATUS", "HAS_STATUS_REASON", "HAS_PHASE", "HAS_STAGE",
        "HAS_TIMELINE", "HAS_SCHEDULE", "HAS_MILESTONE",
        "STARTS_AT", "ENDS_AT", "UPDATED_ON", "RELEASED_ON", "COMPLETES_AT",
        # Organizations
        "OWNED_BY", "PARENT_OF", "HAS_PARENT", "MANAGED_BY", "OPERATED_BY",
        "LED_BY", "RESPONSIBLE_FOR", "WORKS_FOR", "HAS_ROLE",
        "PARTNERED_WITH", "CONTRACTED_BY", "DESIGNED_BY", "ENGINEERED_BY",
        "CONSTRUCTED_BY", "PROCURED_BY", "SUPPLIED_BY", "REGULATED_BY",
        # Geography
        "LOCATED_IN", "HAS_ADDRESS", "IN_CITY", "IN_STATE", "IN_COUNTRY", "IN_REGION", "IN_ZONE",
        # Finance
        "HAS_BUDGET", "HAS_COST", "HAS_CAPEX", "HAS_OPEX", "HAS_TIV", "IN_CURRENCY",
        "FUNDED_BY", "ALLOCATED_TO",
        # Technical
        "IN_INDUSTRY", "IN_SECTOR", "IN_MARKET",
        "PRODUCES", "USES_FEEDSTOCK", "USES_FUEL", "USES_TECHNOLOGY", "USES_PROCESS",
        "REQUIRES_EQUIPMENT", "HAS_UNIT", "HAS_SYSTEM", "HAS_UTILITY", "HAS_CAPACITY",
        "MEETS_STANDARD",
        # Governance
        "REQUIRES_PERMIT", "HAS_REQUIREMENT", "HAS_DELIVERABLE",
        "HAS_ENVIRONMENTAL_ASPECT", "HAS_SAFETY_REQUIREMENT",
        # Risks
        "HAS_RISK", "HAS_ISSUE", "HAS_CHALLENGE", "HAS_CONSTRAINT",
        "CAUSED_BY", "RESULTED_IN", "AFFECTED_BY", "DELAYED_BY", "CANCELLED_DUE_TO",
    ]

    @classmethod
    def get_allowed_nodes(cls) -> List[str]:
        return cls.ALLOWED_NODES.copy()

    @classmethod
    def get_allowed_relationships(cls) -> List[str]:
        return cls.ALLOWED_RELATIONSHIPS.copy()
