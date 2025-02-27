import sqlite3
from typing import Optional, Dict, Any, Union
import psycopg2
from psycopg2 import OperationalError
import pandas as pd
import logging

# Setting up logging for debugging and monitoring.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_connection(
        db_name: str,
        db_type: str,
        host: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
) -> Optional[Union[sqlite3.Connection, psycopg2.extensions.connection]]:
    """
    Creates a connection to the database.
    Supports SQLite, PostgreSQL
    """
    try:
        if db_type.lower() == 'postgresql':
            # Establish a connection to PostgreSQL.
            conn = psycopg2.connect(
                dbname=db_name,
                user=user,
                password=password,
                host=host
            )
            logger.info("Connected to PostgreSQL")
        elif db_type.lower() == 'sqlite':
            # Establish a connection to SQLite.
            conn = sqlite3.connect(db_name)
            logger.info("Connected to SQLite")
        else:
            # Handle unsupported database types.
            logger.error(f"Unsupported database type {db_type}")
            return None
        return conn
    except OperationalError as e:
        # Log database operational errors.
        logger.error(f"Operational error {e}")
    except Exception as e:
        # Catch and log unexpected errors.
        logger.exception(f"Unexpected error {e}")
    return None

def query_database(
        query: str,
        db_name: str,
        db_type: str,
        host: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
) -> pd.DataFrame:
    """
    Executes a SQL query and returns the result as a pandas DataFrame.
    """
    # Create a connection to the database.
    conn = create_connection(db_name, db_type, host, user, password)
    if conn is None:
        logger.error("Database connection failure and returning empty DataFrame")
        return pd.DataFrame()
    try:
        # Execute the query and fetch results into a DataFrame.
        df = pd.read_sql_query(query, conn)
        logger.info("Query executed successfully")
        return df
    except Exception as e:
        # Log errors in query execution.
        logger.error(f"Error executing query: {e}")
        return pd.DataFrame()
    finally:
        # Ensure the database connection is closed.
        conn.close()
        logger.info("Connection closed")

def get_sqlite_table_info(cursor, table_name: str) -> Dict[str, Any]:
    """
    Retrieves metadata for a SQLite table including columns, foreign keys, indexes, and sample data.
    """
    table_info = {'columns': {}, 'foreign_keys': [], 'indexes': [], 'sample_data': []}

    # Retrieve column information.
    cursor.execute(f"PRAGMA table_info(\"{table_name}\");")
    columns = cursor.fetchall()
    for col in columns:
        table_info['columns'][col[1]] = {
            'type': col[2],
            'nullable': not col[3],
            'primary_key': bool(col[5]),
            'default': col[4]
        }

    # Retrieve foreign key information.
    cursor.execute(f"PRAGMA foreign_key_list(\"{table_name}\");")
    fkeys = cursor.fetchall()
    for fk in fkeys:
        table_info['foreign_keys'].append({
            'from_column': fk[3],
            'to_table': fk[2],
            'to_column': fk[4],
            'on_update': fk[5],
            'on_delete': fk[6]
        })

    # Retrieve index information.
    cursor.execute(f"PRAGMA index_list(\"{table_name}\");")
    indexes = cursor.fetchall()
    for idx in indexes:
        cursor.execute(f"PRAGMA index_info(\"{idx[1]}\");")
        index_columns = cursor.fetchall()
        table_info['indexes'].append({
            'name': idx[1],
            'unique': bool(idx[2]),
            'columns': [col[2] for col in index_columns]
        })

    # Retrieve sample data from the table.
    cursor.execute(f"SELECT * FROM \"{table_name}\" LIMIT 5;")
    sample_data = cursor.fetchall()
    if sample_data:
        column_names = [description[0] for description in cursor.description]
        table_info['sample_data'] = [
            dict(zip(column_names, row)) for row in sample_data
        ]

    return table_info

def get_postgresql_table_info(cursor, table_name: str) -> Dict[str, Any]:
    """
    Retrieves metadata for a PostgreSQL table including columns, foreign keys, indexes, and sample data.
    """
    table_info = {'columns': {}, 'foreign_keys': [], 'indexes': [], 'sample_data': []}

    # Retrieve column information.
    cursor.execute("""
        SELECT 
            column_name, 
            data_type,
            is_nullable,
            column_default,
            character_maximum_length
        FROM information_schema.columns 
        WHERE table_name = %s;
    """, [table_name])
    columns = cursor.fetchall()
    for col in columns:
        table_info['columns'][col[0]] = {
            'type': col[1],
            'nullable': col[2] == 'YES',
            'default': col[3],
            'max_length': col[4],
            'primary_key': False  # Will be updated later if it's a primary key.
        }

    # Retrieve primary key information.
    cursor.execute("""
        SELECT c.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.constraint_column_usage AS ccu 
            ON tc.constraint_name = ccu.constraint_name
        JOIN information_schema.columns AS c 
            ON c.table_name = tc.table_name AND c.column_name = ccu.column_name
        WHERE constraint_type = 'PRIMARY KEY' AND tc.table_name = %s;
    """, [table_name])
    pk_columns = [col[0] for col in cursor.fetchall()]
    for col in pk_columns:
        if col in table_info['columns']:
            table_info['columns'][col]['primary_key'] = True

    # Retrieve foreign key information.
    cursor.execute("""
        SELECT
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s;
    """, [table_name])
    fkeys = cursor.fetchall()
    for fk in fkeys:
        table_info['foreign_keys'].append({
            'from_column': fk[0],
            'to_table': fk[1],
            'to_column': fk[2]
        })

    # Retrieve index information.
    cursor.execute("""
        SELECT
            indexname,
            indexdef
        FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = %s;
    """, [table_name])
    indexes = cursor.fetchall()
    for idx_name, idx_def in indexes:
        # Extract columns from index definition.
        idx_columns = idx_def.split('(')[1].rstrip(')').split(', ')
        is_unique = 'UNIQUE' in idx_def.upper()
        table_info['indexes'].append({
            'name': idx_name,
            'unique': is_unique,
            'columns': idx_columns
        })

    # Retrieve sample data from the table.
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
    sample_data = cursor.fetchall()
    if sample_data:
        column_names = [desc[0] for desc in cursor.description]
        table_info['sample_data'] = [
            dict(zip(column_names, row)) for row in sample_data
        ]

    return table_info

def get_all_schemas(
        db_name: str,
        db_type: str,
        host: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieves schema information for all tables in the database.
    """
    # Create a connection to the database.
    conn = create_connection(db_name, db_type, host, user, password)
    if conn is None:
        logger.error("Database connection failed and returning empty schemas")
        return {}
    schemas = {}
    try:
        cursor = conn.cursor()
        if db_type.lower() == 'sqlite':
            # Retrieve all tables in SQLite.
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]

            for table_name in tables:
                schemas[table_name] = get_sqlite_table_info(cursor, table_name)
        elif db_type.lower() == 'postgresql':
            # Retrieve all tables in PostgreSQL.
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            tables = [table[0] for table in cursor.fetchall()]

            for table_name in tables:
                schemas[table_name] = get_postgresql_table_info(cursor, table_name)
        else:
            logger.error(f"Unsupported database type: {db_type}")
            return {}
        logger.info("Schema information retrieved successfully.")
        return schemas

    except Exception as e:
        # Log errors while retrieving schema information.
        logger.exception(f"Error retrieving schema information: {e}")
        return {}

    finally:
        # Ensure the database connection is closed.
        conn.close()
        logger.info("Database connection closed.")
