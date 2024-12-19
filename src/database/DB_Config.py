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
