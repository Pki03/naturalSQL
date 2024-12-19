import sqlite3
from typing import Optional,Dict,Any,Union
import psycopg2
from psycopg2 import OperationalError
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_connection(
        db_name:str,
        db_type:str,
        host:Optional[str]=None,
        user:Optional[str]=None,
        password:Optional[str]=None
)-> Optional[Union[sqlite3.Connection, psycopg2.extensions.connection]]:
    try:
        if db_type.lower() == 'postgresql':
            conn=psycopg2.connect(
                dbname=db_name,
                user=user,
                password=password,
                host=host
            )
            logger.info("connected to postgresql")
        elif db_name.lower() == 'sqlite':
            conn=sqlite3.connect(db_name)
            logger.info("connected to sqlite")
        else:
            logger.error(f"unsupported database type {db_type}")
            return None
        return conn
    except OperationalError as e:
        logger.error(f"operational error {e}")
    except Exception as e:
        logger.exception(f"unexpected error {e}")
    return None

def query_database(
        query:str,
        db_name:str,
        db_type:str,
        host:Optional[str]=None,
        user:Optional[str]=None,
        password:Optional[str]=None
)->pd.DataFrame:
    conn = create_connection(query,db_name,db_type,host,user,password)
    if conn is None:
        logger.error("Database connection failure and returning empty dataframe")
        return pd.DataFrame()
    try:
        df=pd.read_sql_query(query,conn)
        logger.info("query executed successfully")
        return df
    except Exception as e:
        logger.error(f"Error executing {e}")
        return pd.DataFrame()
    finally:
        conn.close()
        logger.info("Connection closed")