import logging
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jConnectionValidator:
    def __init__(self, neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="password"):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None

    def test_connection(self):
        """Test Neo4j database connection and return status"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )

            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]

            logger.info(f"Neo4j connection successful! Test query returned: {test_value}")
            return True, "Connection successful"

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False, str(e)
        finally:
            if self.driver:
                self.driver.close()

    def get_basic_stats(self):
        """Get basic database statistics if connection works"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )

            with self.driver.session() as session:
                # Count nodes
                node_result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = node_result.single()["node_count"]

                # Count relationships
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = rel_result.single()["rel_count"]

                # Get node labels
                labels_result = session.run("CALL db.labels()")
                labels = [record["label"] for record in labels_result]

            stats = {
                "node_count": node_count,
                "relationship_count": rel_count,
                "node_labels": labels
            }

            logger.info(f"Database stats: {stats}")
            return True, stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return False, str(e)
        finally:
            if self.driver:
                self.driver.close()


def main():
    """Test Neo4j connection and display results"""
    validator = Neo4jConnectionValidator()

    print("Testing Neo4j connection...")
    print(f"URI: {validator.neo4j_uri}")
    print(f"User: {validator.neo4j_user}")
    print("-" * 50)

    # Test basic connection
    success, message = validator.test_connection()
    if success:
        print("✓ Connection test PASSED")

        # Get database statistics
        stats_success, stats = validator.get_basic_stats()
        if stats_success:
            print(f"✓ Database statistics:")
            print(f"  - Nodes: {stats['node_count']}")
            print(f"  - Relationships: {stats['relationship_count']}")
            print(f"  - Node labels: {', '.join(stats['node_labels'])}")
        else:
            print(f"✗ Failed to get stats: {stats}")
    else:
        print(f"✗ Connection test FAILED: {message}")


if __name__ == "__main__":
    main()
