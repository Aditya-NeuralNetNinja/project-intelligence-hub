"""
Neo4j database access layer.

Provides centralized Neo4j connectivity and data management
with Aura/hosted instance best practices.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

# LangChain Neo4j integration
try:
    from langchain_community.graphs import Neo4jGraph
except ImportError:
    from langchain.graphs import Neo4jGraph

from src.config import get_logger, log_step
from src.models.project import ProjectRecord, GeoComponents, Milestone
from src.parsers.project_parser import ProjectReportParser

# Module logger
logger = get_logger(__name__)


class Neo4jConnectionError(Exception):
    """Raised when Neo4j connection fails."""
    pass


class Neo4jService:
    """Neo4j access layer with Aura/hosted best practices.

    This class centralizes:
        - Driver construction and connectivity validation
        - LangChain Neo4jGraph wrapper configuration
        - Constraints, structured writes, and database cleanup

    Attributes:
        uri: Neo4j connection URI.
        user: Database username.
        password: Database password.
        database: Database name.
        driver: Low-level Neo4j driver.
        graph: LangChain Neo4jGraph wrapper.

    Raises:
        Neo4jConnectionError: If connection fails.

    Example:
        >>> service = Neo4jService(
        ...     uri="neo4j+s://xxx.databases.neo4j.io",
        ...     user="neo4j",
        ...     password="password"
        ... )
        >>> service.ensure_constraints()
        >>> service.close()
    """

    # Constraint definitions for structured layer
    CONSTRAINTS = [
        "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.projectId IS UNIQUE",
        "CREATE CONSTRAINT project_name IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT budget_key IF NOT EXISTS FOR (b:Budget) REQUIRE b.key IS UNIQUE",
        "CREATE CONSTRAINT location_key IF NOT EXISTS FOR (l:Location) REQUIRE l.key IS UNIQUE",
        "CREATE CONSTRAINT milestone_key IF NOT EXISTS FOR (m:Milestone) REQUIRE m.key IS UNIQUE",
        "CREATE CONSTRAINT report_key IF NOT EXISTS FOR (r:Report) REQUIRE r.key IS UNIQUE",
    ]

    # Performance indexes for faster queries
    INDEXES = [
        "CREATE INDEX project_name_idx IF NOT EXISTS FOR (p:Project) ON (p.name)",
        "CREATE INDEX project_source_idx IF NOT EXISTS FOR (p:Project) ON (p.source)",
        "CREATE INDEX chunk_source_idx IF NOT EXISTS FOR (c:Chunk) ON (c.source)",
        "CREATE INDEX milestone_date_idx IF NOT EXISTS FOR (m:Milestone) ON (m.dateText)",
        "CREATE INDEX location_city_idx IF NOT EXISTS FOR (l:Location) ON (l.city)",
        "CREATE INDEX location_country_idx IF NOT EXISTS FOR (l:Location) ON (l.country)",
        "CREATE INDEX challenge_source_idx IF NOT EXISTS FOR (c:Challenge) ON (c.source)",
    ]

    # Full-text index for semantic search within graph
    FULLTEXT_INDEX = """
    CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
    FOR (n:Project|Organization|Location|Milestone|Challenge)
    ON EACH [n.name, n.text, n.description]
    """

    # Cypher template with APOC support
    # Uses CALL subqueries to handle empty lists properly
    CYPHER_UPSERT_WITH_APOC = """
    MERGE (p:Project {projectId: $project_id})
      ON CREATE SET p.name = $project_name
      ON MATCH SET p.name = coalesce(p.name, $project_name)
    SET p.source = $source,
        p.status = $status,
        p.statusReason = $status_reason,
        p.lastUpdate = $last_update,
        p.initialRelease = $initial_release

    WITH p
    MERGE (b:Budget {key: $bud_key})
    SET b.amount = $tiv_amount,
        b.currency = $tiv_currency,
        b.kind = 'TIV',
        b.source = $source
    MERGE (p)-[:HAS_BUDGET]->(b)

    WITH p
    MERGE (l:Location {key: $loc_key})
    SET l.address = $address,
        l.city = $city,
        l.state = $state,
        l.postal = $postal,
        l.country = $country,
        l.zoneCounty = $zone_county,
        l.source = $source
    MERGE (p)-[:LOCATED_IN]->(l)

    WITH p
    MERGE (r:Report {key: $rep_key})
    SET r.source = $source,
        r.lastUpdate = $last_update,
        r.initialRelease = $initial_release
    MERGE (p)-[:HAS_REPORT]->(r)

    WITH p
    CALL {
        WITH p
        UNWIND CASE WHEN size($challenges) > 0 THEN $challenges ELSE [null] END AS ch
        WITH p, ch WHERE ch IS NOT NULL
        MERGE (c:Challenge {key: p.projectId + '::ch::' + toString(apoc.util.md5(ch))})
        SET c.text = ch, c.source = $source
        MERGE (p)-[:HAS_CHALLENGE]->(c)
        RETURN count(*) AS chCount
    }

    WITH p
    CALL {
        WITH p
        UNWIND CASE WHEN size($milestones) > 0 THEN $milestones ELSE [null] END AS ms
        WITH p, ms WHERE ms IS NOT NULL
        MERGE (m:Milestone {key: p.projectId + '::ms::' + toString(apoc.util.md5(ms.sentence))})
        SET m.name = ms.name, m.dateText = ms.dateText, m.sentence = ms.sentence, m.source = $source
        MERGE (p)-[:HAS_MILESTONE]->(m)
        RETURN count(*) AS msCount
    }

    RETURN p.projectId AS projectId, p.name AS name
    """

    # Cypher template without APOC (fallback)
    # Uses CALL subqueries to handle empty lists properly
    CYPHER_UPSERT_NO_APOC = """
    MERGE (p:Project {projectId: $project_id})
      ON CREATE SET p.name = $project_name
      ON MATCH SET p.name = coalesce(p.name, $project_name)
    SET p.source = $source,
        p.status = $status,
        p.statusReason = $status_reason,
        p.lastUpdate = $last_update,
        p.initialRelease = $initial_release

    WITH p
    MERGE (b:Budget {key: $bud_key})
    SET b.amount = $tiv_amount,
        b.currency = $tiv_currency,
        b.kind = 'TIV',
        b.source = $source
    MERGE (p)-[:HAS_BUDGET]->(b)

    WITH p
    MERGE (l:Location {key: $loc_key})
    SET l.address = $address,
        l.city = $city,
        l.state = $state,
        l.postal = $postal,
        l.country = $country,
        l.zoneCounty = $zone_county,
        l.source = $source
    MERGE (p)-[:LOCATED_IN]->(l)

    WITH p
    MERGE (r:Report {key: $rep_key})
    SET r.source = $source,
        r.lastUpdate = $last_update,
        r.initialRelease = $initial_release
    MERGE (p)-[:HAS_REPORT]->(r)

    WITH p
    CALL {
        WITH p
        UNWIND CASE WHEN size($challenges) > 0 THEN range(0, size($challenges)-1) ELSE [null] END AS i
        WITH p, i WHERE i IS NOT NULL
        MERGE (c:Challenge {key: p.projectId + '::ch::' + toString(i)})
        SET c.text = $challenges[i], c.source = $source
        MERGE (p)-[:HAS_CHALLENGE]->(c)
        RETURN count(*) AS chCount
    }

    WITH p
    CALL {
        WITH p
        UNWIND CASE WHEN size($milestones) > 0 THEN range(0, size($milestones)-1) ELSE [null] END AS j
        WITH p, j WHERE j IS NOT NULL
        MERGE (m:Milestone {key: p.projectId + '::ms::' + toString(j)})
        SET m.name = $milestones[j].name, m.dateText = $milestones[j].dateText,
            m.sentence = $milestones[j].sentence, m.source = $source
        MERGE (p)-[:HAS_MILESTONE]->(m)
        RETURN count(*) AS msCount
    }

    RETURN p.projectId AS projectId, p.name AS name
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j"
    ) -> None:
        """Initialize Neo4j service.

        Args:
            uri: Neo4j URI (typically neo4j+s://... for Aura).
            user: Neo4j username.
            password: Neo4j password.
            database: Neo4j database name (Aura commonly uses "neo4j").

        Raises:
            Neo4jConnectionError: If connection or authentication fails.
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database or "neo4j"

        logger.info(f"Connecting to Neo4j: {uri}")
        try:
            # Low-level driver for constraint management and transactional writes
            logger.substep("Creating driver")
            self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.substep("Driver connectivity verified")

            # LangChain wrapper for GraphCypherQAChain and graph operations
            logger.substep("Initializing Neo4jGraph wrapper")
            self.graph: Neo4jGraph = Neo4jGraph(
                url=uri,
                username=user,
                password=password,
                database=self.database
            )
            logger.info(f"Connected to Neo4j database: {self.database}")
        except ServiceUnavailable as e:
            logger.error(f"Service unavailable: {e}")
            raise Neo4jConnectionError(
                f"Could not connect to Neo4j at {uri}. "
                f"Ensure the URI is correct and the database is running. "
                f"Error: {e}"
            ) from e
        except AuthError as e:
            logger.error(f"Authentication failed: {e}")
            raise Neo4jConnectionError(
                f"Authentication failed for Neo4j. "
                f"Check username and password. Error: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise Neo4jConnectionError(
                f"Failed to connect to Neo4j: {e}"
            ) from e

        self._parser = ProjectReportParser()

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        logger.debug("Closing Neo4j driver")
        try:
            self.driver.close()
            logger.debug("Neo4j driver closed")
        except Exception as e:
            logger.warning(f"Error closing driver: {e}")

    def ensure_constraints(self) -> None:
        """Create constraints for the structured layer.

        Notes:
            Some Aura tiers or policies may restrict certain DDL operations.
            Failures are logged but swallowed to keep ingestion operational.
        """
        with log_step(logger, "Create database constraints"):
            success_count = 0
            with self.driver.session(database=self.database) as session:
                for stmt in self.CONSTRAINTS:
                    try:
                        session.run(stmt)
                        success_count += 1
                    except Exception as e:
                        logger.debug(f"Constraint skipped: {e}")
            logger.info(f"Constraints created: {success_count}/{len(self.CONSTRAINTS)}")

        # Also create performance indexes
        self.ensure_indexes()

    def ensure_indexes(self) -> None:
        """Create performance indexes for faster queries.

        Creates indexes on frequently queried properties and
        optionally a full-text index for semantic search.
        """
        with log_step(logger, "Create performance indexes"):
            success_count = 0
            with self.driver.session(database=self.database) as session:
                for stmt in self.INDEXES:
                    try:
                        session.run(stmt)
                        success_count += 1
                    except Exception as e:
                        logger.debug(f"Index skipped: {e}")

                # Try to create full-text index (may not be available on all tiers)
                try:
                    session.run(self.FULLTEXT_INDEX)
                    logger.substep("Full-text index created")
                except Exception as e:
                    logger.debug(f"Full-text index skipped: {e}")

            logger.info(f"Indexes created: {success_count}/{len(self.INDEXES)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics for monitoring.

        Returns:
            Dictionary with node/relationship counts and other stats.
        """
        stats: Dict[str, Any] = {}

        queries = {
            "node_count": "MATCH (n) RETURN count(n) AS count",
            "relationship_count": "MATCH ()-[r]->() RETURN count(r) AS count",
            "project_count": "MATCH (p:Project) RETURN count(p) AS count",
            "chunk_count": "MATCH (c:Chunk) RETURN count(c) AS count",
            "entity_count": "MATCH (e) WHERE NOT e:Chunk AND NOT e:Project RETURN count(e) AS count",
        }

        for name, query in queries.items():
            try:
                result = self.graph.query(query)
                stats[name] = result[0]["count"] if result else 0
            except Exception:
                stats[name] = -1

        return stats

    def clear(self) -> None:
        """Delete all nodes and relationships from the database."""
        logger.info("Clearing all nodes and relationships from database")
        self.graph.query("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")

    def upsert_structured_project(
        self,
        record: ProjectRecord
    ) -> Dict[str, Any]:
        """Upsert structured nodes/relationships for a single project record.

        This function is the reliability backbone for:
            - Budget allocation & location questions
            - Timeline comparison questions
            - Challenges questions (derived from reason/details/schedule heuristics)

        Args:
            record: Parsed ProjectRecord.

        Returns:
            Dictionary with {"projectId": ..., "name": ...}.
        """
        project_name = record.project_name or record.source
        logger.debug(f"Upserting project: {project_name}")

        project_key = record.get_unique_key()
        loc_key = f"{project_key}::loc"
        bud_key = f"{project_key}::tiv"
        rep_key = f"{project_key}::report::{record.last_update or ''}"

        # Parse geographic components
        geo = self._parser.parse_city_state_country(record.city_state_line)

        # Derive challenges and milestones
        challenges = self._parser.derive_challenges(record)
        milestones = self._parser.extract_milestones(record.schedule_text)
        milestone_dicts = [m.to_dict() for m in milestones]

        logger.substep(f"Extracted {len(challenges)} challenges, {len(milestones)} milestones")
        if milestones:
            for ms in milestones:
                logger.substep(f"  Milestone: {ms.name} -> {ms.date_text}")
        else:
            logger.warning(f"No milestones extracted from schedule_text: {record.schedule_text[:100] if record.schedule_text else 'None'}...")

        params = {
            # Identification
            "source": record.source,
            "project_id": record.project_id or record.project_name or record.source,
            "project_name": record.project_name or record.source,
            # Classification
            "industry_code": record.industry_code,
            "project_type": record.project_type,
            "sector": record.sector,
            "sic_code": record.sic_code,
            # Financial
            "bud_key": bud_key,
            "tiv_amount": record.tiv_amount,
            "tiv_currency": record.tiv_currency,
            # Status
            "status": record.status,
            "status_reason": record.status_reason,
            "project_probability": record.project_probability,
            # Timeline
            "last_update": record.last_update,
            "initial_release": record.initial_release,
            "pec_timing": record.pec_timing,
            "pec_activity": record.pec_activity,
            # Location
            "loc_key": loc_key,
            "address": record.address,
            "city": geo.city,
            "state": geo.state,
            "postal": geo.postal,
            "country": geo.country,
            "zone_county": record.zone_county,
            "phone": record.phone,
            # Plant Info
            "plant_owner": record.plant_owner,
            "plant_parent": record.plant_parent,
            "plant_name": record.plant_name,
            "plant_id": record.plant_id,
            "unit_name": record.unit_name,
            # Contacts
            "project_manager": record.project_manager,
            "project_manager_company": record.project_manager_company,
            "project_manager_email": record.project_manager_email,
            "engineer_company": record.engineer_company,
            "ec_firm": record.ec_firm,
            # Technical
            "scope_text": record.scope_text,
            "project_capacity": record.project_capacity,
            "environmental": record.environmental,
            "construction_labor": record.construction_labor,
            "fuel_type": record.fuel_type,
            # Report
            "rep_key": rep_key,
            # Derived
            "challenges": challenges,
            "milestones": milestone_dicts,
        }

        with self.driver.session(database=self.database) as session:
            # Step 1: Upsert base project with all fields
            base_query = """
            MERGE (p:Project {projectId: $project_id})
              ON CREATE SET p.name = $project_name
              ON MATCH SET p.name = coalesce(p.name, $project_name)
            SET p.source = $source,
                // Classification
                p.industryCode = $industry_code,
                p.projectType = $project_type,
                p.sector = $sector,
                p.sicCode = $sic_code,
                // Status
                p.status = $status,
                p.statusReason = $status_reason,
                p.projectProbability = $project_probability,
                // Timeline
                p.lastUpdate = $last_update,
                p.initialRelease = $initial_release,
                p.pecTiming = $pec_timing,
                p.pecActivity = $pec_activity,
                // Plant Info
                p.plantOwner = $plant_owner,
                p.plantParent = $plant_parent,
                p.plantName = $plant_name,
                p.plantId = $plant_id,
                p.unitName = $unit_name,
                p.phone = $phone,
                // Contacts
                p.projectManager = $project_manager,
                p.projectManagerCompany = $project_manager_company,
                p.projectManagerEmail = $project_manager_email,
                p.engineerCompany = $engineer_company,
                p.ecFirm = $ec_firm,
                // Technical
                p.scopeText = $scope_text,
                p.projectCapacity = $project_capacity,
                p.environmental = $environmental,
                p.constructionLabor = $construction_labor,
                p.fuelType = $fuel_type

            WITH p
            MERGE (b:Budget {key: $bud_key})
            SET b.amount = $tiv_amount, b.currency = $tiv_currency, b.kind = 'TIV', b.source = $source
            MERGE (p)-[:HAS_BUDGET]->(b)

            WITH p
            MERGE (l:Location {key: $loc_key})
            SET l.address = $address, l.city = $city, l.state = $state,
                l.postal = $postal, l.country = $country, l.zoneCounty = $zone_county, l.source = $source
            MERGE (p)-[:LOCATED_IN]->(l)

            WITH p
            MERGE (r:Report {key: $rep_key})
            SET r.source = $source, r.lastUpdate = $last_update, r.initialRelease = $initial_release
            MERGE (p)-[:HAS_REPORT]->(r)

            RETURN p.projectId AS projectId, p.name AS name
            """
            logger.substep("Executing base project upsert")
            row = session.run(base_query, params).single()

            if row is None:
                logger.warning("Base project upsert returned no result")
                return {"projectId": params["project_id"], "name": params["project_name"]}

            project_id = row["projectId"]
            project_name = row["name"]
            logger.substep(f"Project created: {project_name}")

            # Step 2: Add challenges (separate query)
            if challenges:
                for i, ch in enumerate(challenges):
                    ch_query = """
                    MATCH (p:Project {projectId: $project_id})
                    MERGE (c:Challenge {key: $ch_key})
                    SET c.text = $ch_text, c.source = $source
                    MERGE (p)-[:HAS_CHALLENGE]->(c)
                    """
                    session.run(ch_query, {
                        "project_id": project_id,
                        "ch_key": f"{project_id}::ch::{i}",
                        "ch_text": ch,
                        "source": record.source
                    })
                logger.substep(f"Added {len(challenges)} challenges")

            # Step 3: Add milestones (separate query)
            if milestone_dicts:
                for i, ms in enumerate(milestone_dicts):
                    ms_query = """
                    MATCH (p:Project {projectId: $project_id})
                    MERGE (m:Milestone {key: $ms_key})
                    SET m.name = $ms_name, m.dateText = $ms_date, m.sentence = $ms_sentence, m.source = $source
                    MERGE (p)-[:HAS_MILESTONE]->(m)
                    """
                    session.run(ms_query, {
                        "project_id": project_id,
                        "ms_key": f"{project_id}::ms::{i}",
                        "ms_name": ms.get("name", ""),
                        "ms_date": ms.get("dateText", ""),
                        "ms_sentence": ms.get("sentence", ""),
                        "source": record.source
                    })
                logger.substep(f"Added {len(milestone_dicts)} milestones")

            return {"projectId": project_id, "name": project_name}

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results.

        Args:
            cypher: Cypher query string.
            params: Optional query parameters.

        Returns:
            List of result dictionaries.
        """
        return self.graph.query(cypher, params or {})

    def __enter__(self) -> "Neo4jService":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()
