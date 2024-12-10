import argparse
import concurrent
import datetime
import pymysql.cursors
import pandas as pd
import requests
import base64
import os
import numpy as np
import json
from retrying import retry
from sqlalchemy import create_engine, event, exc
from sqlalchemy.engine import URL
from sqlalchemy.pool import QueuePool


# import pyspark.pandas as ps
MAX_WORKERS = 6
STOP_MAX_ATTEMPT_NUMBER = 6
WAIT_EXPONENTIAL_MULTIPLIER = 3000

LOCAL_SCHEMA_ETL = 'etl'
LOCAL_TABLE_RUN_DATA = 'DWD_POC_CASE_EES_RUN_DATA_ALL'
LOCAL_TABLE_RUN_DATA_CONDITION = 'DWD_POC_CASE_WAFER_CONDITION5'
LOCAL_TABLE_RUN_DATA_UNPIVOT = 'DWD_POC_CASE_EES_RUN_DATA_UNPIVOT_STEP'
LOCAL_CACHE_TABLE_RUN_DATA_UNPIVOT = 'M_DWD_POC_CASE_EES_RUN_DATA_UNPIVOT_STEP'
LOCAL_TABLE_TOOL_STEP_INFO = 'M_CASE_DISTINCT_TOOL_STEP_INFO_LOCAL'
LOCAL_TABLE_TOOL_PARAMETRIC_INFO = 'DWD_POC_TOOL_PARAMETRIC_INFO'
LOCAL_CACHE_TABLE_TOOL_PARAMETRIC_INFO = 'M_DWD_POC_TOOL_PARAMETRIC_INFO'
LOCAL_TABLE_PRODUCTS_MAP = 'DWD_POC_PRODUCTS_MAP'
LOCAL_TABLE_TOOL_EQP_MAP = 'DWD_POC_TOOL_EQP_MAP'
LOCAL_TABLE_TOOL_PRODUCT_OPER_MAP = 'DWD_POC_TOOL_PRODUCT_OPER_MAP'
LOCAL_TABLE_STATS_RESULTS = 'stats_results_all'
LOCAL_TABLE_STATS_RESULTS_LOG = 'stats_results_log'
LOCAL_TABLE_ETL_LOG = 'DWD_CASE_ETL_LOG'


def __create_engine(host, port, user, password, database):
    url = URL.create(
        "mysql+pymysql",
        username = user,
        password = password,
        host = host,
        port = port,
        database = database
    )
    #print(url)
    return create_engine(url, poolclass=QueuePool, pool_size=MAX_WORKERS, max_overflow=5)


def init_online_engine():
    return __create_engine('10.52.199.91', '9030', 'ikas_user', 'Ikas_user@123', 'ODS_E3SUITE')


def init_local_engine():
    return __create_engine(args.host, args.port, args.user, args.password, args.database)


def read_as_df(sql, engine, index_column):
    try:
        with engine.connect() as conn:
            sql_data = pd.read_sql(sql, conn, coerce_float=False)
            sql_data.columns = index_column
            #print(sql_data)
            return sql_data
    except Exception as e:
        print("error")
        print(e)
        raise e


def df_to_doris(df, table_name, engine, index=True, columns=None):
    if df is not None and len(df) > 0:
        with engine.connect() as conn:
            try:
                print(f"=================== insert into {table_name} ================")
                df.to_sql(table_name, con=conn, if_exists='append', index=index, index_label=columns)
            except Exception as e:
                print(e)


def doris_stream_load_from_df(df, engine, table, is_json=True, chunksize=100000, partitions=None):
    engine_url = engine.url
    url = 'http://%s:18030/api/%s/%s/_stream_load' % (engine_url.host, engine_url.database, table)

    format_str = 'csv' if not is_json else 'json'
    headers = {
        'Content-Type': 'text/plain; charset=UTF-8',
        'format': format_str,
        'Expect': '100-continue'
    }
    if is_json:
        headers['strip_outer_array'] = 'true'
        headers['read_json_by_line'] = 'true'
    else:
        headers['column_separator'] = '@'
    
    if partitions:
        headers['partitions'] = partitions
    
    auth = requests.auth.HTTPBasicAuth(engine_url.username, engine_url.password)
    session = requests.sessions.Session()
    session.should_strip_auth = lambda old_url, new_url: False
    
    l = len(df)
    if l > 0:
        if chunksize and chunksize < l:
            batches = l // chunksize
            if l % chunksize > 0:
                batches += 1
            for i in range(batches):
                si = i * chunksize
                ei = min(si + chunksize, l)
                sub = df[si:ei]
                do_doris_stream_load_from_df(sub, session, url, headers, auth, is_json)
        else:
            do_doris_stream_load_from_df(df, session, url, headers, auth, is_json)


def do_doris_stream_load_from_df(df, session, url, headers, auth, is_json=False):
    data = df.to_csv(header=False, index=False, sep='@') if not is_json else df.to_json(orient='records', date_format='iso')
    #print(data)
    
    resp = session.request(
        'PUT',
        url = url,
        data=data.encode('utf-8'),
        headers=headers,
        auth=auth
    )
    print(resp.reason, resp.text)
    check_stream_load_response(resp.text)


def check_stream_load_response(resp_text):
    resp = json.loads(resp_text)
    if resp['Status'] not in ["Success", "Publish Timeout"]:
        raise Exception(resp['Message'])


def read_online_products_map():
    sql = "SELECT DISTINCT PRODUCT_ID, PRODG1 FROM ODS_EDA.ODS_PRODUCT;"
    print("=================== read_online_products_map ================")
    return read_as_df(sql, online_engine, ['PRODUCT_ID', 'PRODG1'])


def read_online_tool_eqp_map():
    sql = (f"select d1.ID TOOL_ID, d1.NAME TOOL_NAME, d2.ID EQP_ID, d2.NAME EQP_NAME "
           f"from "
           f"(SELECT ID, NAME, PARENT_TOOL_ID FROM ODS_E3SUITE.ODS_EES_TOOL WHERE PARENT_TOOL_ID !=0) d1 "
           f"INNER JOIN "
           f"(SELECT ID, NAME, PARENT_TOOL_ID FROM ODS_E3SUITE.ODS_EES_TOOL WHERE PARENT_TOOL_ID =0) d2 "
           f"on d1.PARENT_TOOL_ID = d2.ID")
    print("=================== read_online_tool_eqp_map ================")
    return read_as_df(sql, online_engine, ['TOOL_ID', 'TOOL_NAME', 'EQP_ID', 'EQP_NAME'])


def read_online_distinct_tools(case_name, min_start_time, max_start_time):
    #sql = (f"select TOOL_ID, '{case_name}' case_info, ROW_NUMBER() over(ORDER BY TOOL_ID) rn "
    #       f"from ("
    #       f"select distinct TOOL_ID "
    #       f"from ODS_E3SUITE.ODS_EES_RUN_CONTEXT partition {case_name_to_doris_partition_name(case_name)} d1 "
    #       f"where WAFER_ID is not null and WAFER_ID <> '' and WAFER_ID <> 'NA' "
    #       f"and START_TIME >= '{min_start_time}' and START_TIME < '{max_start_time}'"
    #       f") a;")
    sql = (f"select TOOL_ID, '{case_name}' case_info, ROW_NUMBER() over(ORDER BY TOOL_ID) rn "
           f"from ("
           f"select distinct d1.TOOL_ID "
           f"from ODS_E3SUITE.ODS_EES_RUN_CONTEXT partition {case_name_to_doris_partition_name(case_name)} d1 "
           f"join ODS_E3SUITE.ODS_EES_FD_UVA_HIST partition {case_name_to_doris_partition_name(case_name)} d2 "
           f"on d1.TOOL_ID=d2.TOOL_ID and d1.RUN_ID=d2.RUN_ID "
           f"where d1.WAFER_ID is not null and d1.WAFER_ID <> '' and d1.WAFER_ID <> 'NA' "
           f") a;")
    print("=================== read_online_distinct_tools ================")
    print(sql)
    online_engine.dispose()
    return read_as_df(sql, online_engine, ['tool_id', 'case_info', 'rn'])


def read_online_product_oper_map(case_name):
    sql = (
        f"select d.MODULE, d.SECTIONS, d.TOOL_ID, d.TOOL_NAME, d.COLUMN_NAME, c.COLUMN_NAME from("
        f"SELECT a.MODULE, a.SECTIONS, a.TOOL_ID, a.TOOL_NAME, a.COLUMN_NAME from("
        f"SELECT ortti.MODULE, ortti.SECTIONS, ortti.TOOL_ID, ortti.TOOL_NAME, ortptv.COLUMN_NAME FROM ODS_E3SUITE.ODS_EES_TOOL oet "
        f"inner join ODS_E3RPT.ODS_RPT_TBL_EQP_TEMPL_CHAMBER_DEF ortetcd "
        f"on oet.EQUIP_TEMPL_CHMBR_ID = ortetcd.ID "
        f"INNER JOIN ODS_E3RPT.ODS_RPT_TBL_TOOL_INFO ortti "
        f"on oet.ID = ortti.TOOL_ID "
        f"INNER JOIN ODS_E3RPT.ODS_RPT_TBL_PROC_TYPE_VAR ortptv "
        f"on ortetcd.PROC_TYPE_ID = ortptv.PROC_TYPE_ID "
        f"WHERE ortptv.NAME = 'PRODUCT_ID' and ortptv.TABLE_NAME = 'EES_RUN_CONTEXT' ) a) d "
        f"inner join "
        f"(SELECT b.MODULE, b.SECTIONS, b.TOOL_ID, b.TOOL_NAME, b.COLUMN_NAME from( "
        f"SELECT ortti.MODULE, ortti.SECTIONS, ortti.TOOL_ID, ortti.TOOL_NAME, ortptv.COLUMN_NAME FROM ODS_E3SUITE.ODS_EES_TOOL oet "
        f"inner join ODS_E3RPT.ODS_RPT_TBL_EQP_TEMPL_CHAMBER_DEF ortetcd "
        f"on oet.EQUIP_TEMPL_CHMBR_ID = ortetcd.ID "
        f"INNER JOIN ODS_E3RPT.ODS_RPT_TBL_TOOL_INFO ortti "
        f"on oet.ID = ortti.TOOL_ID "
        f"INNER JOIN ODS_E3RPT.ODS_RPT_TBL_PROC_TYPE_VAR ortptv "
        f"on ortetcd.PROC_TYPE_ID = ortptv.PROC_TYPE_ID "
        f"WHERE ortptv.NAME = 'OPER_NO' and ortptv.TABLE_NAME = 'EES_RUN_CONTEXT' ) b) c "
        f"on d.TOOL_ID=c.TOOL_ID and d.TOOL_ID IS NOT NULL;"
    )
    print("=================== read_online_product_oper_map ================")
    #print(sql)
    return read_as_df(sql, online_engine, ['MODULE', 'SECTIONS', 'TOOL_ID', 'TOOL_NAME', 'PRODUCT_ID_COLUMMN', 'OPER_NO_COLUMMN'])


def assemble_online_context_sql(case_name, product_id, oper_no, tool_id, min_start_time, max_start_time):
    sql = (
        f"select d1.TOOL_ID, d1.PARTITION_ID, d1.RUN_ID, d1.START_TIME, d1.END_TIME, d1.RECIPE_NAME, d1.BATCH_ID, d1.CARRIER_ID, d1.LOT_ID, d1.WAFER_ID, d1.SLOT_NUMBER, d1.DATA_QUALITY"
        f", d1.CONTEXT_1, d1.CONTEXT_2, CONTEXT_3, CONTEXT_4, CONTEXT_5, CONTEXT_6, CONTEXT_7, CONTEXT_8, CONTEXT_9, CONTEXT_10"
        f", CONTEXT_11, CONTEXT_12, CONTEXT_13, CONTEXT_14, CONTEXT_15, CONTEXT_16, CONTEXT_17, CONTEXT_18, CONTEXT_19, CONTEXT_20"
        f", CONTEXT_21, CONTEXT_22, CONTEXT_23, CONTEXT_24, CONTEXT_25, CONTEXT_26, CONTEXT_27, CONTEXT_28, CONTEXT_29, CONTEXT_30"
        f", CONTEXT_31, CONTEXT_32, CONTEXT_33, CONTEXT_34, CONTEXT_35, CONTEXT_36, CONTEXT_37, CONTEXT_38, CONTEXT_39, CONTEXT_40"
        f", CONTEXT_41, CONTEXT_42, CONTEXT_43, CONTEXT_44, CONTEXT_45, CONTEXT_46, CONTEXT_47, CONTEXT_48, CONTEXT_49, CONTEXT_50"
        f", '{case_name}' CASE_INFO, d1.{product_id} PRODUCT_ID, d1.{oper_no} OPER_NO "
        f"from ODS_E3SUITE.ODS_EES_RUN_CONTEXT partition {case_name_to_doris_partition_name(case_name)} d1 "
        f"where d1.WAFER_ID is not null and d1.WAFER_ID <> '' and d1.WAFER_ID <> 'NA' "
        f"and d1.START_TIME >= '{min_start_time}' and d1.START_TIME < '{max_start_time}' "
        f"and d1.TOOL_ID = {tool_id}"
    )
    print("=================== assemble_online_context_sql ================")
    print(sql)
    return sql


def read_online_context(case_name, product_id, oper_no, tool_id, min_start_time, max_start_time):
    print("=================== read_online_context ================")
    sql = assemble_online_context_sql(case_name, product_id, oper_no, tool_id, min_start_time, max_start_time)
    columns = [
        'TOOL_ID',
        'PARTITION_ID',
        'RUN_ID',
        'START_TIME',
        'END_TIME',
        'RECIPE_NAME',
        'BATCH_ID',
        'CARRIER_ID',
        'LOT_ID',
        'WAFER_ID',
        'SLOT_NUMBER',
        'DATA_QUALITY',
        'CONTEXT_1',
        'CONTEXT_2', 'CONTEXT_3', 'CONTEXT_4', 'CONTEXT_5', 'CONTEXT_6', 'CONTEXT_7', 'CONTEXT_8', 'CONTEXT_9', 'CONTEXT_10',
        'CONTEXT_11', 'CONTEXT_12', 'CONTEXT_13', 'CONTEXT_14', 'CONTEXT_15', 'CONTEXT_16', 'CONTEXT_17', 'CONTEXT_18', 'CONTEXT_19', 'CONTEXT_20',
        'CONTEXT_21', 'CONTEXT_22', 'CONTEXT_23', 'CONTEXT_24', 'CONTEXT_25', 'CONTEXT_26', 'CONTEXT_27', 'CONTEXT_28', 'CONTEXT_29', 'CONTEXT_30',
        'CONTEXT_31', 'CONTEXT_32', 'CONTEXT_33', 'CONTEXT_34', 'CONTEXT_35', 'CONTEXT_36', 'CONTEXT_37', 'CONTEXT_38', 'CONTEXT_39', 'CONTEXT_40',
        'CONTEXT_41', 'CONTEXT_42', 'CONTEXT_43', 'CONTEXT_44', 'CONTEXT_45', 'CONTEXT_46', 'CONTEXT_47', 'CONTEXT_48', 'CONTEXT_49', 'CONTEXT_50',
        'CASE_INFO',
        'PRODUCT_ID',
        'OPER_NO'
    ]
    return read_as_df(sql, online_engine, columns)


def read_online_products():
    print("=================== read_online_products ================")
    sql = "SELECT DISTINCT PRODUCT_ID FROM ODS_EDA.ODS_PRODUCT WHERE PRODCAT_ID ='Production' AND PRODG1 LIKE 'L55%%' OR PRODG1 LIKE 'L40%%' OR PRODG1 LIKE 'L28%%';"
    online_engine.dispose()
    return read_as_df(sql, online_engine, ['PRODUCT_ID'])


def format_as_product_in_conditions(df):
    productIds = df['PRODUCT_ID'].values.tolist()
    return "({})".format(','.join([f"'{str(t)}'" for t in productIds]))


def read_online_data(case_name, product_id, oper_no, tool_id, min_start_time, max_start_time, target_products=None):
    #sql = (f"select a.TOOL_ID"
    #       f",a.PARTITION_ID"
    #       f",a.TIME_STAMP"
    #       f",a.START_TIME"
    #       f",a.RUN_ID"
    #       f",'{case_name}' CASE_INFO"
    #       f",a.PRODUCT_ID"
    #       f",a.OPER_NO"
    #       f",a.LOT_ID"
    #       f",a.WAFER_ID"
    #       f",a.RECIPE_NAME"
    #       f",a.EVENT_NAME"
    #       f",a.EVENT_TYPE"
    #       f",a.ALARM_ID"
    #       f",a.ALARM_CODE"
    #       f",a.VAR1,a.VAR2,a.VAR3,a.VAR4,a.VAR5,a.VAR6,a.VAR7,a.VAR8,a.VAR9,a.VAR10"
    #       f",a.VAR11,a.VAR12,a.VAR13,a.VAR14,a.VAR15,a.VAR16,a.VAR17,a.VAR18,a.VAR19,a.VAR20"
    #       f",a.VAR21,a.VAR22,a.VAR23,a.VAR24,a.VAR25,a.VAR26,a.VAR27,a.VAR28,a.VAR29,a.VAR30"
    #       f",a.VAR31,a.VAR32,a.VAR33,a.VAR34,a.VAR35,a.VAR36,a.VAR37,a.VAR38,a.VAR39,a.VAR40"
    #       f",a.VAR41,a.VAR42,a.VAR43,a.VAR44,a.VAR45,a.VAR46,a.VAR47,a.VAR48,a.VAR49,a.VAR50"
    #       f",a.VAR51,a.VAR52,a.VAR53,a.VAR54,a.VAR55,a.VAR56,a.VAR57,a.VAR58,a.VAR59,a.VAR60"
    #       f",a.VAR61,a.VAR62,a.VAR63,a.VAR64,a.VAR65,a.VAR66,a.VAR67,a.VAR68,a.VAR69,a.VAR70"
    #       f",a.VAR71,a.VAR72,a.VAR73,a.VAR74,a.VAR75,a.VAR76,a.VAR77,a.VAR78,a.VAR79,a.VAR80"
    #       f",a.VAR81,a.VAR82,a.VAR83,a.VAR84,a.VAR85,a.VAR86,a.VAR87,a.VAR88,a.VAR89,a.VAR90"
    #       f",a.VAR91,a.VAR92,a.VAR93,a.VAR94,a.VAR95,a.VAR96,a.VAR97,a.VAR98,a.VAR99,a.VAR100"
    #       f",a.VAR101,a.VAR102,a.VAR103,a.VAR104,a.VAR105,a.VAR106,a.VAR107,a.VAR108,a.VAR109,a.VAR110"
    #       f",a.VAR111,a.VAR112,a.VAR113,a.VAR114,a.VAR115,a.VAR116,a.VAR117,a.VAR118,a.VAR119,a.VAR120"
    #       f",a.VAR121,a.VAR122,a.VAR123,a.VAR124,a.VAR125,a.VAR126,a.VAR127,a.VAR128,a.VAR129,a.VAR130"
    #       f",a.VAR131,a.VAR132,a.VAR133,a.VAR134,a.VAR135,a.VAR136,a.VAR137,a.VAR138,a.VAR139,a.VAR140"
    #       f",a.VAR141,a.VAR142,a.VAR143,a.VAR144,a.VAR145,a.VAR146,a.VAR147,a.VAR148,a.VAR149,a.VAR150"
    #       f",a.VAR151,a.VAR152,a.VAR153,a.VAR154,a.VAR155,a.VAR156,a.VAR157,a.VAR158,a.VAR159,a.VAR160"
    #       f",a.VAR161,a.VAR162,a.VAR163,a.VAR164,a.VAR165,a.VAR166,a.VAR167,a.VAR168,a.VAR169,a.VAR170"
    #       f",a.VAR171,a.VAR172,a.VAR173,a.VAR174,a.VAR175,a.VAR176,a.VAR177,a.VAR178,a.VAR179,a.VAR180"
    #       f",a.VAR181,a.VAR182,a.VAR183,a.VAR184,a.VAR185,a.VAR186,a.VAR187,a.VAR188,a.VAR189,a.VAR190"
    #       f",a.VAR191,a.VAR192,a.VAR193,a.VAR194,a.VAR195,a.VAR196,a.VAR197,a.VAR198,a.VAR199,a.VAR200 "
    #       f"from ("
    #       f"select d2.{product_id} PRODUCT_ID,d2.{oper_no} OPER_NO,d2.LOT_ID,d2.WAFER_ID,d2.RECIPE_NAME"
    #       f",row_number() over(partition by d1.TOOL_ID,d1.RUN_ID,d1.TIME_STAMP order by d1.TIME_STAMP) rn"
    #       f",d1.* "
    #       f"from ODS_E3SUITE.ODS_EES_RUN_DATA partition {case_name_to_doris_partition_name(case_name)} d1 "
    #       f"join ODS_E3SUITE.ODS_EES_RUN_CONTEXT partition {case_name_to_doris_partition_name(case_name)} d2 "
    #       f"on d1.tool_id=d2.tool_id and d1.run_id=d2.run_id "
    #       f"where d1.START_TIME >= '{min_start_time}' and d1.START_TIME < '{max_start_time}' and d1.tool_id={tool_id} "
    #       f"and d2.WAFER_ID is not null and d2.WAFER_ID <> '' and d2.WAFER_ID <> 'NA'")
    #if target_products is not None and len(target_products) > 0:
    #    sql += f" and d2.{product_id} in {format_as_product_in_conditions(target_products)}"
    #sql += ") a where rn=1;"
    
    sql = (f"select d1.TOOL_ID"
           f",d1.PARTITION_ID"
           f",d1.TIME_STAMP"
           f",d1.START_TIME"
           f",d1.RUN_ID"
           f",'{case_name}' CASE_INFO"
           f",d2.{product_id} PRODUCT_ID,d2.{oper_no} OPER_NO,d2.LOT_ID,d2.WAFER_ID,d2.RECIPE_NAME"
           f",d1.EVENT_NAME"
           f",d1.EVENT_TYPE"
           f",d1.ALARM_ID"
           f",d1.ALARM_CODE"
           f",d1.VAR1,d1.VAR2,d1.VAR3,d1.VAR4,d1.VAR5,d1.VAR6,d1.VAR7,d1.VAR8,d1.VAR9,d1.VAR10"
           f",d1.VAR11,d1.VAR12,d1.VAR13,d1.VAR14,d1.VAR15,d1.VAR16,d1.VAR17,d1.VAR18,d1.VAR19,d1.VAR20"
           f",d1.VAR21,d1.VAR22,d1.VAR23,d1.VAR24,d1.VAR25,d1.VAR26,d1.VAR27,d1.VAR28,d1.VAR29,d1.VAR30"
           f",d1.VAR31,d1.VAR32,d1.VAR33,d1.VAR34,d1.VAR35,d1.VAR36,d1.VAR37,d1.VAR38,d1.VAR39,d1.VAR40"
           f",d1.VAR41,d1.VAR42,d1.VAR43,d1.VAR44,d1.VAR45,d1.VAR46,d1.VAR47,d1.VAR48,d1.VAR49,d1.VAR50"
           f",d1.VAR51,d1.VAR52,d1.VAR53,d1.VAR54,d1.VAR55,d1.VAR56,d1.VAR57,d1.VAR58,d1.VAR59,d1.VAR60"
           f",d1.VAR61,d1.VAR62,d1.VAR63,d1.VAR64,d1.VAR65,d1.VAR66,d1.VAR67,d1.VAR68,d1.VAR69,d1.VAR70"
           f",d1.VAR71,d1.VAR72,d1.VAR73,d1.VAR74,d1.VAR75,d1.VAR76,d1.VAR77,d1.VAR78,d1.VAR79,d1.VAR80"
           f",d1.VAR81,d1.VAR82,d1.VAR83,d1.VAR84,d1.VAR85,d1.VAR86,d1.VAR87,d1.VAR88,d1.VAR89,d1.VAR90"
           f",d1.VAR91,d1.VAR92,d1.VAR93,d1.VAR94,d1.VAR95,d1.VAR96,d1.VAR97,d1.VAR98,d1.VAR99,d1.VAR100"
           f",d1.VAR101,d1.VAR102,d1.VAR103,d1.VAR104,d1.VAR105,d1.VAR106,d1.VAR107,d1.VAR108,d1.VAR109,d1.VAR110"
           f",d1.VAR111,d1.VAR112,d1.VAR113,d1.VAR114,d1.VAR115,d1.VAR116,d1.VAR117,d1.VAR118,d1.VAR119,d1.VAR120"
           f",d1.VAR121,d1.VAR122,d1.VAR123,d1.VAR124,d1.VAR125,d1.VAR126,d1.VAR127,d1.VAR128,d1.VAR129,d1.VAR130"
           f",d1.VAR131,d1.VAR132,d1.VAR133,d1.VAR134,d1.VAR135,d1.VAR136,d1.VAR137,d1.VAR138,d1.VAR139,d1.VAR140"
           f",d1.VAR141,d1.VAR142,d1.VAR143,d1.VAR144,d1.VAR145,d1.VAR146,d1.VAR147,d1.VAR148,d1.VAR149,d1.VAR150"
           f",d1.VAR151,d1.VAR152,d1.VAR153,d1.VAR154,d1.VAR155,d1.VAR156,d1.VAR157,d1.VAR158,d1.VAR159,d1.VAR160"
           f",d1.VAR161,d1.VAR162,d1.VAR163,d1.VAR164,d1.VAR165,d1.VAR166,d1.VAR167,d1.VAR168,d1.VAR169,d1.VAR170"
           f",d1.VAR171,d1.VAR172,d1.VAR173,d1.VAR174,d1.VAR175,d1.VAR176,d1.VAR177,d1.VAR178,d1.VAR179,d1.VAR180"
           f",d1.VAR181,d1.VAR182,d1.VAR183,d1.VAR184,d1.VAR185,d1.VAR186,d1.VAR187,d1.VAR188,d1.VAR189,d1.VAR190"
           f",d1.VAR191,d1.VAR192,d1.VAR193,d1.VAR194,d1.VAR195,d1.VAR196,d1.VAR197,d1.VAR198,d1.VAR199,d1.VAR200 "
           f"from ODS_E3SUITE.ODS_EES_RUN_DATA partition {case_name_to_doris_partition_name(case_name)} d1 "
           f"join ODS_E3SUITE.ODS_EES_RUN_CONTEXT partition {case_name_to_doris_partition_name(case_name)} d2 "
           f"on d1.TOOL_ID=d2.TOOL_ID and d1.RUN_ID=d2.RUN_ID "
           f"where d1.START_TIME >= '{min_start_time}' and d1.START_TIME < '{max_start_time}' and d1.TOOL_ID={tool_id} "
           f"and d2.WAFER_ID is not null and d2.WAFER_ID <> '' and d2.WAFER_ID <> 'NA'")
    if target_products is not None and len(target_products) > 0:
        sql += f" and d2.{product_id} in {format_as_product_in_conditions(target_products)}"
    sql += ";"
    
    print("=================== read_online_data ================")
    print(sql)
    columns = [
        "TOOL_ID",
        "PARTITION_ID",
        "TIME_STAMP",
        "START_TIME",
        "RUN_ID",
        "CASE_INFO",
        "PRODUCT_ID",
        "OPER_NO",
        "LOT_ID",
        "WAFER_ID",
        "RECIPE_NAME",
        "EVENT_NAME",
        "EVENT_TYPE",
        "ALARM_ID",
        "ALARM_CODE",
        "VAR1","VAR2","VAR3","VAR4","VAR5","VAR6","VAR7","VAR8","VAR9","VAR10",
        "VAR11","VAR12","VAR13","VAR14","VAR15","VAR16","VAR17","VAR18","VAR19","VAR20",
        "VAR21","VAR22","VAR23","VAR24","VAR25","VAR26","VAR27","VAR28","VAR29","VAR30",
        "VAR31","VAR32","VAR33","VAR34","VAR35","VAR36","VAR37","VAR38","VAR39","VAR40",
        "VAR41","VAR42","VAR43","VAR44","VAR45","VAR46","VAR47","VAR48","VAR49","VAR50",
        "VAR51","VAR52","VAR53","VAR54","VAR55","VAR56","VAR57","VAR58","VAR59","VAR60",
        "VAR61","VAR62","VAR63","VAR64","VAR65","VAR66","VAR67","VAR68","VAR69","VAR70",
        "VAR71","VAR72","VAR73","VAR74","VAR75","VAR76","VAR77","VAR78","VAR79","VAR80",
        "VAR81","VAR82","VAR83","VAR84","VAR85","VAR86","VAR87","VAR88","VAR89","VAR90",
        "VAR91","VAR92","VAR93","VAR94","VAR95","VAR96","VAR97","VAR98","VAR99","VAR100",
        "VAR101","VAR102","VAR103","VAR104","VAR105","VAR106","VAR107","VAR108","VAR109","VAR110",
        "VAR111","VAR112","VAR113","VAR114","VAR115","VAR116","VAR117","VAR118","VAR119","VAR120",
        "VAR121","VAR122","VAR123","VAR124","VAR125","VAR126","VAR127","VAR128","VAR129","VAR130",
        "VAR131","VAR132","VAR133","VAR134","VAR135","VAR136","VAR137","VAR138","VAR139","VAR140",
        "VAR141","VAR142","VAR143","VAR144","VAR145","VAR146","VAR147","VAR148","VAR149","VAR150",
        "VAR151","VAR152","VAR153","VAR154","VAR155","VAR156","VAR157","VAR158","VAR159","VAR160",
        "VAR161","VAR162","VAR163","VAR164","VAR165","VAR166","VAR167","VAR168","VAR169","VAR170",
        "VAR171","VAR172","VAR173","VAR174","VAR175","VAR176","VAR177","VAR178","VAR179","VAR180",
        "VAR181","VAR182","VAR183","VAR184","VAR185","VAR186","VAR187","VAR188","VAR189","VAR190",
        "VAR191","VAR192","VAR193","VAR194","VAR195","VAR196","VAR197","VAR198","VAR199","VAR200"
    ]
    online_engine.dispose()
    return read_as_df(sql, online_engine, columns)


def read_online_data_with_tools(case_name, product_id, oper_no, tool_ids, min_start_time, max_start_time, target_products=None):
    # 当前数据库里TIME_STAMP是有毫秒数的，根据TIME_STAMP来做row number不起作用
    #sql = (f"select a.TOOL_ID"
    #       f",a.PARTITION_ID"
    #       f",a.TIME_STAMP"
    #       f",a.START_TIME"
    #       f",a.RUN_ID"
    #       f",'{case_name}' CASE_INFO"
    #       f",a.PRODUCT_ID"
    #       f",a.OPER_NO"
    #       f",a.LOT_ID"
    #       f",a.WAFER_ID"
    #       f",a.RECIPE_NAME"
    #       f",a.EVENT_NAME"
    #       f",a.EVENT_TYPE"
    #       f",a.ALARM_ID"
    #       f",a.ALARM_CODE"
    #       f",a.VAR1,a.VAR2,a.VAR3,a.VAR4,a.VAR5,a.VAR6,a.VAR7,a.VAR8,a.VAR9,a.VAR10"
    #       f",a.VAR11,a.VAR12,a.VAR13,a.VAR14,a.VAR15,a.VAR16,a.VAR17,a.VAR18,a.VAR19,a.VAR20"
    #       f",a.VAR21,a.VAR22,a.VAR23,a.VAR24,a.VAR25,a.VAR26,a.VAR27,a.VAR28,a.VAR29,a.VAR30"
    #       f",a.VAR31,a.VAR32,a.VAR33,a.VAR34,a.VAR35,a.VAR36,a.VAR37,a.VAR38,a.VAR39,a.VAR40"
    #       f",a.VAR41,a.VAR42,a.VAR43,a.VAR44,a.VAR45,a.VAR46,a.VAR47,a.VAR48,a.VAR49,a.VAR50"
    #       f",a.VAR51,a.VAR52,a.VAR53,a.VAR54,a.VAR55,a.VAR56,a.VAR57,a.VAR58,a.VAR59,a.VAR60"
    #       f",a.VAR61,a.VAR62,a.VAR63,a.VAR64,a.VAR65,a.VAR66,a.VAR67,a.VAR68,a.VAR69,a.VAR70"
    #       f",a.VAR71,a.VAR72,a.VAR73,a.VAR74,a.VAR75,a.VAR76,a.VAR77,a.VAR78,a.VAR79,a.VAR80"
    #       f",a.VAR81,a.VAR82,a.VAR83,a.VAR84,a.VAR85,a.VAR86,a.VAR87,a.VAR88,a.VAR89,a.VAR90"
    #       f",a.VAR91,a.VAR92,a.VAR93,a.VAR94,a.VAR95,a.VAR96,a.VAR97,a.VAR98,a.VAR99,a.VAR100"
    #       f",a.VAR101,a.VAR102,a.VAR103,a.VAR104,a.VAR105,a.VAR106,a.VAR107,a.VAR108,a.VAR109,a.VAR110"
    #       f",a.VAR111,a.VAR112,a.VAR113,a.VAR114,a.VAR115,a.VAR116,a.VAR117,a.VAR118,a.VAR119,a.VAR120"
    #       f",a.VAR121,a.VAR122,a.VAR123,a.VAR124,a.VAR125,a.VAR126,a.VAR127,a.VAR128,a.VAR129,a.VAR130"
    #       f",a.VAR131,a.VAR132,a.VAR133,a.VAR134,a.VAR135,a.VAR136,a.VAR137,a.VAR138,a.VAR139,a.VAR140"
    #       f",a.VAR141,a.VAR142,a.VAR143,a.VAR144,a.VAR145,a.VAR146,a.VAR147,a.VAR148,a.VAR149,a.VAR150"
    #       f",a.VAR151,a.VAR152,a.VAR153,a.VAR154,a.VAR155,a.VAR156,a.VAR157,a.VAR158,a.VAR159,a.VAR160"
    #       f",a.VAR161,a.VAR162,a.VAR163,a.VAR164,a.VAR165,a.VAR166,a.VAR167,a.VAR168,a.VAR169,a.VAR170"
    #       f",a.VAR171,a.VAR172,a.VAR173,a.VAR174,a.VAR175,a.VAR176,a.VAR177,a.VAR178,a.VAR179,a.VAR180"
    #       f",a.VAR181,a.VAR182,a.VAR183,a.VAR184,a.VAR185,a.VAR186,a.VAR187,a.VAR188,a.VAR189,a.VAR190"
    #       f",a.VAR191,a.VAR192,a.VAR193,a.VAR194,a.VAR195,a.VAR196,a.VAR197,a.VAR198,a.VAR199,a.VAR200 "
    #       f"from ("
    #       f"select d2.{product_id} PRODUCT_ID,d2.{oper_no} OPER_NO,d2.LOT_ID,d2.WAFER_ID,d2.RECIPE_NAME"
    #       f",row_number() over(partition by d1.TOOL_ID,d1.RUN_ID,d1.TIME_STAMP order by d1.TIME_STAMP) rn"
    #       f",d1.* "
    #       f"from ODS_E3SUITE.ODS_EES_RUN_DATA partition {case_name_to_doris_partition_name(case_name)} d1 "
    #       f"join ODS_E3SUITE.ODS_EES_RUN_CONTEXT partition {case_name_to_doris_partition_name(case_name)} d2 "
    #       f"on d1.TOOL_ID=d2.TOOL_ID and d1.RUN_ID=d2.RUN_ID "
    #       f"where d1.START_TIME >= '{min_start_time}' and d1.START_TIME < '{max_start_time}' and d1.TOOL_ID in {tool_ids} "
    #       f"and d2.WAFER_ID is not null and d2.WAFER_ID <> '' and d2.WAFER_ID <> 'NA'")
    #if target_products is not None and len(target_products) > 0:
    #    sql += f" and d2.{product_id} in {format_as_product_in_conditions(target_products)}"
    #sql += ") a where rn=1;"
    
    sql = (f"select d1.TOOL_ID"
           f",d1.PARTITION_ID"
           f",d1.TIME_STAMP"
           f",d1.START_TIME"
           f",d1.RUN_ID"
           f",'{case_name}' CASE_INFO"
           f",d2.{product_id} PRODUCT_ID,d2.{oper_no} OPER_NO,d2.LOT_ID,d2.WAFER_ID,d2.RECIPE_NAME"
           f",d1.EVENT_NAME"
           f",d1.EVENT_TYPE"
           f",d1.ALARM_ID"
           f",d1.ALARM_CODE"
           f",d1.VAR1,d1.VAR2,d1.VAR3,d1.VAR4,d1.VAR5,d1.VAR6,d1.VAR7,d1.VAR8,d1.VAR9,d1.VAR10"
           f",d1.VAR11,d1.VAR12,d1.VAR13,d1.VAR14,d1.VAR15,d1.VAR16,d1.VAR17,d1.VAR18,d1.VAR19,d1.VAR20"
           f",d1.VAR21,d1.VAR22,d1.VAR23,d1.VAR24,d1.VAR25,d1.VAR26,d1.VAR27,d1.VAR28,d1.VAR29,d1.VAR30"
           f",d1.VAR31,d1.VAR32,d1.VAR33,d1.VAR34,d1.VAR35,d1.VAR36,d1.VAR37,d1.VAR38,d1.VAR39,d1.VAR40"
           f",d1.VAR41,d1.VAR42,d1.VAR43,d1.VAR44,d1.VAR45,d1.VAR46,d1.VAR47,d1.VAR48,d1.VAR49,d1.VAR50"
           f",d1.VAR51,d1.VAR52,d1.VAR53,d1.VAR54,d1.VAR55,d1.VAR56,d1.VAR57,d1.VAR58,d1.VAR59,d1.VAR60"
           f",d1.VAR61,d1.VAR62,d1.VAR63,d1.VAR64,d1.VAR65,d1.VAR66,d1.VAR67,d1.VAR68,d1.VAR69,d1.VAR70"
           f",d1.VAR71,d1.VAR72,d1.VAR73,d1.VAR74,d1.VAR75,d1.VAR76,d1.VAR77,d1.VAR78,d1.VAR79,d1.VAR80"
           f",d1.VAR81,d1.VAR82,d1.VAR83,d1.VAR84,d1.VAR85,d1.VAR86,d1.VAR87,d1.VAR88,d1.VAR89,d1.VAR90"
           f",d1.VAR91,d1.VAR92,d1.VAR93,d1.VAR94,d1.VAR95,d1.VAR96,d1.VAR97,d1.VAR98,d1.VAR99,d1.VAR100"
           f",d1.VAR101,d1.VAR102,d1.VAR103,d1.VAR104,d1.VAR105,d1.VAR106,d1.VAR107,d1.VAR108,d1.VAR109,d1.VAR110"
           f",d1.VAR111,d1.VAR112,d1.VAR113,d1.VAR114,d1.VAR115,d1.VAR116,d1.VAR117,d1.VAR118,d1.VAR119,d1.VAR120"
           f",d1.VAR121,d1.VAR122,d1.VAR123,d1.VAR124,d1.VAR125,d1.VAR126,d1.VAR127,d1.VAR128,d1.VAR129,d1.VAR130"
           f",d1.VAR131,d1.VAR132,d1.VAR133,d1.VAR134,d1.VAR135,d1.VAR136,d1.VAR137,d1.VAR138,d1.VAR139,d1.VAR140"
           f",d1.VAR141,d1.VAR142,d1.VAR143,d1.VAR144,d1.VAR145,d1.VAR146,d1.VAR147,d1.VAR148,d1.VAR149,d1.VAR150"
           f",d1.VAR151,d1.VAR152,d1.VAR153,d1.VAR154,d1.VAR155,d1.VAR156,d1.VAR157,d1.VAR158,d1.VAR159,d1.VAR160"
           f",d1.VAR161,d1.VAR162,d1.VAR163,d1.VAR164,d1.VAR165,d1.VAR166,d1.VAR167,d1.VAR168,d1.VAR169,d1.VAR170"
           f",d1.VAR171,d1.VAR172,d1.VAR173,d1.VAR174,d1.VAR175,d1.VAR176,d1.VAR177,d1.VAR178,d1.VAR179,d1.VAR180"
           f",d1.VAR181,d1.VAR182,d1.VAR183,d1.VAR184,d1.VAR185,d1.VAR186,d1.VAR187,d1.VAR188,d1.VAR189,d1.VAR190"
           f",d1.VAR191,d1.VAR192,d1.VAR193,d1.VAR194,d1.VAR195,d1.VAR196,d1.VAR197,d1.VAR198,d1.VAR199,d1.VAR200 "
           f"from ODS_E3SUITE.ODS_EES_RUN_DATA partition {case_name_to_doris_partition_name(case_name)} d1 "
           f"join ODS_E3SUITE.ODS_EES_RUN_CONTEXT partition {case_name_to_doris_partition_name(case_name)} d2 "
           f"on d1.TOOL_ID=d2.TOOL_ID and d1.RUN_ID=d2.RUN_ID "
           f"where d1.START_TIME >= '{min_start_time}' and d1.START_TIME < '{max_start_time}' and d1.TOOL_ID in {tool_ids} "
           f"and d2.WAFER_ID is not null and d2.WAFER_ID <> '' and d2.WAFER_ID <> 'NA'")
    if target_products is not None and len(target_products) > 0:
        sql += f" and d2.{product_id} in {format_as_product_in_conditions(target_products)}"
    sql += ";"
    
    print("=================== read_online_data_with_tools ================")
    print(sql)
    columns = [
        "TOOL_ID",
        "PARTITION_ID",
        "TIME_STAMP",
        "START_TIME",
        "RUN_ID",
        "CASE_INFO",
        "PRODUCT_ID",
        "OPER_NO",
        "LOT_ID",
        "WAFER_ID",
        "RECIPE_NAME",
        "EVENT_NAME",
        "EVENT_TYPE",
        "ALARM_ID",
        "ALARM_CODE",
        "VAR1","VAR2","VAR3","VAR4","VAR5","VAR6","VAR7","VAR8","VAR9","VAR10",
        "VAR11","VAR12","VAR13","VAR14","VAR15","VAR16","VAR17","VAR18","VAR19","VAR20",
        "VAR21","VAR22","VAR23","VAR24","VAR25","VAR26","VAR27","VAR28","VAR29","VAR30",
        "VAR31","VAR32","VAR33","VAR34","VAR35","VAR36","VAR37","VAR38","VAR39","VAR40",
        "VAR41","VAR42","VAR43","VAR44","VAR45","VAR46","VAR47","VAR48","VAR49","VAR50",
        "VAR51","VAR52","VAR53","VAR54","VAR55","VAR56","VAR57","VAR58","VAR59","VAR60",
        "VAR61","VAR62","VAR63","VAR64","VAR65","VAR66","VAR67","VAR68","VAR69","VAR70",
        "VAR71","VAR72","VAR73","VAR74","VAR75","VAR76","VAR77","VAR78","VAR79","VAR80",
        "VAR81","VAR82","VAR83","VAR84","VAR85","VAR86","VAR87","VAR88","VAR89","VAR90",
        "VAR91","VAR92","VAR93","VAR94","VAR95","VAR96","VAR97","VAR98","VAR99","VAR100",
        "VAR101","VAR102","VAR103","VAR104","VAR105","VAR106","VAR107","VAR108","VAR109","VAR110",
        "VAR111","VAR112","VAR113","VAR114","VAR115","VAR116","VAR117","VAR118","VAR119","VAR120",
        "VAR121","VAR122","VAR123","VAR124","VAR125","VAR126","VAR127","VAR128","VAR129","VAR130",
        "VAR131","VAR132","VAR133","VAR134","VAR135","VAR136","VAR137","VAR138","VAR139","VAR140",
        "VAR141","VAR142","VAR143","VAR144","VAR145","VAR146","VAR147","VAR148","VAR149","VAR150",
        "VAR151","VAR152","VAR153","VAR154","VAR155","VAR156","VAR157","VAR158","VAR159","VAR160",
        "VAR161","VAR162","VAR163","VAR164","VAR165","VAR166","VAR167","VAR168","VAR169","VAR170",
        "VAR171","VAR172","VAR173","VAR174","VAR175","VAR176","VAR177","VAR178","VAR179","VAR180",
        "VAR181","VAR182","VAR183","VAR184","VAR185","VAR186","VAR187","VAR188","VAR189","VAR190",
        "VAR191","VAR192","VAR193","VAR194","VAR195","VAR196","VAR197","VAR198","VAR199","VAR200"
    ]
    online_engine.dispose()
    return read_as_df(sql, online_engine, columns)
    

def read_online_data_condition_with_tools(case_name, product_id, oper_no, tool_ids, min_start_time, max_start_time, target_products=None):
    #sql = (f"select distinct d2.WAFER_ID,d2.LOT_ID,d2.{product_id} PRODUCT_ID,d3.PRODG1,d1.TOOL_ID,d4.TOOL_NAME,d5.EQP_ID,d5.EQP_NAME,d2.{oper_no} OPER_NO,d2.RECIPE_NAME,d1.START_TIME "
    #       f"from ODS_E3SUITE.ODS_EES_RUN_DATA partition {case_name_to_doris_partition_name(case_name)} d1 "
    #       f"join ODS_E3SUITE.ODS_EES_RUN_CONTEXT partition {case_name_to_doris_partition_name(case_name)} d2 "
    #       f"on d1.TOOL_ID=d2.TOOL_ID and d1.RUN_ID=d2.RUN_ID "
    #       f"join (SELECT DISTINCT PRODUCT_ID, PRODG1, PRODCAT_ID FROM ODS_EDA.ODS_PRODUCT) d3 on d2.{product_id} = d3.PRODUCT_ID "
    #       f"join ODS_E3RPT.ODS_RPT_TBL_TOOL_INFO d4 on d1.TOOL_ID=d4.TOOL_ID "
    #       f"join ("
    #       f"select d1.ID TOOL_ID, d1.NAME TOOL_NAME, d2.ID EQP_ID, d2.NAME EQP_NAME "
    #       f"from "
    #       f"(SELECT ID, NAME, PARENT_TOOL_ID FROM ODS_E3SUITE.ODS_EES_TOOL WHERE PARENT_TOOL_ID !=0) d1 "
    #       f"INNER JOIN "
    #       f"(SELECT ID, NAME, PARENT_TOOL_ID FROM ODS_E3SUITE.ODS_EES_TOOL WHERE PARENT_TOOL_ID =0) d2 "
    #       f"on d1.PARENT_TOOL_ID = d2.ID"
    #       f") d5 on d1.TOOL_ID=d5.TOOL_ID "
    #       f"where d1.START_TIME >= '{min_start_time}' and d1.START_TIME < '{max_start_time}' and d1.TOOL_ID in {tool_ids} "
    #       f"and d2.WAFER_ID is not null and d2.WAFER_ID <> '' and d2.WAFER_ID <> 'NA'")
    sql = (f"select distinct d2.WAFER_ID,d2.LOT_ID,d2.{product_id} PRODUCT_ID,d3.PRODG1,d5.TOOL_ID,d5.TOOL_NAME,d5.EQP_ID,d5.EQP_NAME,d2.{oper_no} OPER_NO,d2.RECIPE_NAME,d2.START_TIME "
           f"from ODS_E3SUITE.ODS_EES_RUN_CONTEXT partition {case_name_to_doris_partition_name(case_name)} d2 "
           f"join (SELECT DISTINCT PRODUCT_ID, PRODG1, PRODCAT_ID FROM ODS_EDA.ODS_PRODUCT) d3 on d2.{product_id} = d3.PRODUCT_ID "
           f"join ("
           f"select d1.ID TOOL_ID, d1.NAME TOOL_NAME, d2.ID EQP_ID, d2.NAME EQP_NAME "
           f"from "
           f"(SELECT ID, NAME, PARENT_TOOL_ID FROM ODS_E3SUITE.ODS_EES_TOOL WHERE PARENT_TOOL_ID !=0) d1 "
           f"INNER JOIN "
           f"(SELECT ID, NAME, PARENT_TOOL_ID FROM ODS_E3SUITE.ODS_EES_TOOL WHERE PARENT_TOOL_ID =0) d2 "
           f"on d1.PARENT_TOOL_ID = d2.ID"
           f") d5 on d2.TOOL_ID=d5.TOOL_ID "
           f"where d2.START_TIME >= '{min_start_time}' and d2.START_TIME < '{max_start_time}' and d2.TOOL_ID in {tool_ids} "
           f"and d2.WAFER_ID is not null and d2.WAFER_ID <> '' and d2.WAFER_ID <> 'NA'")
    if target_products is not None and len(target_products) > 0:
        sql += f" and d2.{product_id} in {format_as_product_in_conditions(target_products)}"
    sql += ";"
    
    print("=================== read_online_data_condition_with_tools ================")
    print(sql)
    columns = [
        "WAFER_ID",
        "LOT_ID",
        "PRODUCT_ID",
        "PRODG1",
        "TOOL_ID",
        "TOOL_NAME",
        "EQP_ID",
        "EQP_NAME",
        "OPER_NO",
        "RECIPE_NAME",
        "START_TIME"
    ]
    return read_as_df(sql, online_engine, columns)


def read_online_uva_data(case_name, product_id, oper_no, tool_id, min_start_time, max_start_time):
    sql = (f"select d1.TOOL_ID"
           f",d5.TOOL_NAME"
           f",d1.RUN_ID"
           f",d6.EQP_NAME"
           f",'{case_name}' CASE_INFO"
           f",d2.{product_id} PRODUCT_ID"
           f",d4.PRODG1"
           f",d2.{oper_no} OPER_NO"
           f",d2.LOT_ID"
           f",d2.WAFER_ID"
           f",d2.RECIPE_NAME"
           f",UPPER(d3.INPUT)"
           f",UPPER(d3.WINDOW)"
           f",UPPER(d3.STATISTICS) STATISTICS"
           f",UPPER(CONCAT(d3.INPUT, '#', d3.WINDOW, '#', d3.STATISTICS)) parametric_name"
           f",d1.START_TIME"
           f",d1.PARTITION_ID"
           f",d1.STATISTIC_KEY"
           f",d1.COLLECTION_KEY"
           f",d1.SEQ_ID"
           f",d1.PROCESS_TYPE_ID"
           f",d1.TIME_STAMP"
           f",d1.CALC_TIME_STAMP"
           f",d1.TARGET"
           f",d1.LOWER_WARNING"
           f",d1.UPPER_WARNING"
           f",d1.LOWER_CRITICAL"
           f",d1.UPPER_CRITICAL"
           f",d1.LOWER_OUTLIER"
           f",d1.UPPER_OUTLIER"
           f",d1.RULES_ENABLED"
           f",d1.ALARM_RULE"
           f",d1.RESULT"
           f",d1.STATUS"
           f",d1.REGION"
           f",d1.ERROR_MSG"
           f",d1.STATISTIC_RESULT"
           f",d1.VERSION "
           f"from ODS_E3SUITE.ODS_EES_FD_UVA_HIST partition {case_name_to_doris_partition_name(case_name)} d1 "
           f"join ODS_E3SUITE.ODS_EES_RUN_CONTEXT partition {case_name_to_doris_partition_name(case_name)} d2 "
           f"on d1.TOOL_ID=d2.TOOL_ID and d1.RUN_ID=d2.RUN_ID "
           f"join ("
           f"select b.SEQ_ID,b.COLLECTION_NAME,a.COLLECTION_KEY,a.COLLECTION_TYPE,b.STATISTICS"
           f",b.`INPUT`,b.`WINDOW`,b.CONTEXT_GROUP "
           f"from ODS_E3RPT.ODS_RPT_TBL_COLLECTION a "
           f"join ODS_E3RPT.ODS_RPT_TBL_CONFIG_DETAILS b "
           f"on a.name=b.COLLECTION_NAME"
           f") d3 "
           f"on d1.COLLECTION_KEY=d3.COLLECTION_KEY and d1.SEQ_ID=d3.SEQ_ID "
           f"join(SELECT DISTINCT PRODUCT_ID, PRODG1, PRODCAT_ID FROM ODS_EDA.ODS_PRODUCT) d4 on d2.{product_id} = d4.PRODUCT_ID "
           f"join ODS_E3RPT.ODS_RPT_TBL_TOOL_INFO d5 on d1.TOOL_ID=d5.TOOL_ID "
           f"join ("
           f"select d1.ID TOOL_ID, d1.NAME TOOL_NAME, d2.ID EQP_ID, d2.NAME EQP_NAME "
           f"from "
           f"(SELECT ID, NAME, PARENT_TOOL_ID FROM ODS_E3SUITE.ODS_EES_TOOL WHERE PARENT_TOOL_ID !=0) d1 "
           f"INNER JOIN "
           f"(SELECT ID, NAME, PARENT_TOOL_ID FROM ODS_E3SUITE.ODS_EES_TOOL WHERE PARENT_TOOL_ID =0) d2 "
           f"on d1.PARENT_TOOL_ID = d2.ID"
           f") d6 "
           f"on d1.TOOL_ID = d6.TOOL_ID "
           f"where d1.START_TIME >= '{min_start_time}' and d1.START_TIME < '{max_start_time}' "
           f"and d1.RULES_ENABLED=1 and d1.TOOL_ID={tool_id} and d2.WAFER_ID is not null and d2.WAFER_ID <> '' and d2.WAFER_ID <> 'NA' and d4.PRODCAT_ID ='Production';")
    print("=================== read_online_uva_data ================")
    print(sql)
    columns = [
        "TOOL_ID",
        "TOOL_NAME",
        "RUN_ID",
        "EQP_NAME",
        "CASE_INFO",
        "PRODUCT_ID",
        "PRODG1",
        "OPER_NO",
        "LOT_ID",
        "WAFER_ID",
        "RECIPE_NAME",
        "INPUT",
        "WINDOW",
        "STATISTICS",
        "parametric_name",
        "START_TIME",
        "PARTITION_ID",
        "STATISTIC_KEY",
        "COLLECTION_KEY",
        "SEQ_ID",
        "PROCESS_TYPE_ID",
        "TIME_STAMP",
        "CALC_TIME_STAMP",
        "TARGET",
        "LOWER_WARNING",
        "UPPER_WARNING",
        "LOWER_CRITICAL",
        "UPPER_CRITICAL",
        "LOWER_OUTLIER",
        "UPPER_OUTLIER",
        "RULES_ENABLED",
        "ALARM_RULE",
        "RESULT",
        "STATUS",
        "REGION",
        "ERROR_MSG",
        "STATISTIC_RESULT",
        "VERSION"
    ]
    return read_as_df(sql, online_engine, columns)


def read_online_uva_data_new(case_name, product_id, oper_no, tool_id, min_start_time, max_start_time):
    sql = (f"select d1.TOOL_ID"
           f",d5.TOOL_NAME"
           f",d1.RUN_ID"
           f",d6.EQP_NAME"
           f",'{case_name}' CASE_INFO"
           f",d2.{product_id} PRODUCT_ID"
           f",d4.PRODG1"
           f",d2.{oper_no} OPER_NO"
           f",d2.LOT_ID"
           f",d2.WAFER_ID"
           f",d2.RECIPE_NAME"
           f",UPPER(d3.INPUT)"
           f",UPPER(d3.WINDOW)"
           f",UPPER(d3.STATISTICS) STATISTICS"
           f",UPPER(CONCAT(d3.INPUT, '#', d3.WINDOW, '#', d3.STATISTICS)) parametric_name"
           f",d1.START_TIME"
           f",d1.PARTITION_ID"
           f",d1.STATISTIC_KEY"
           f",d1.COLLECTION_KEY"
           f",d1.SEQ_ID"
           f",d1.PROCESS_TYPE_ID"
           f",d1.TIME_STAMP"
           f",d1.CALC_TIME_STAMP"
           f",d1.TARGET"
           f",d1.LOWER_WARNING"
           f",d1.UPPER_WARNING"
           f",d1.LOWER_CRITICAL"
           f",d1.UPPER_CRITICAL"
           f",d1.LOWER_OUTLIER"
           f",d1.UPPER_OUTLIER"
           f",d1.RULES_ENABLED"
           f",d1.ALARM_RULE"
           f",d1.RESULT"
           f",d1.STATUS"
           f",d1.REGION"
           f",d1.ERROR_MSG"
           f",d1.STATISTIC_RESULT"
           f",d1.VERSION "
           f"from ODS_E3SUITE.ODS_EES_FD_UVA_HIST partition {case_name_to_doris_partition_name(case_name)} d1 "
           f"join ODS_E3SUITE.ODS_EES_RUN_CONTEXT partition {case_name_to_doris_partition_name(case_name)} d2 "
           f"on d1.TOOL_ID=d2.TOOL_ID and d1.RUN_ID=d2.RUN_ID "
           f"join DWD_DEVELOP.COLLECTION_WITH_CONTEXT d3 "
           f"on d1.COLLECTION_KEY=d3.COLLECTION_KEY and d1.SEQ_ID=d3.SEQ_ID "
           f"join (SELECT DISTINCT PRODUCT_ID, PRODG1, PRODCAT_ID FROM ODS_EDA.ODS_PRODUCT) d4 on d2.{product_id} = d4.PRODUCT_ID "
           f"join ODS_E3RPT.ODS_RPT_TBL_TOOL_INFO d5 on d1.TOOL_ID=d5.TOOL_ID "
           f"join DWD_DEVELOP.TOOL_EQP_MAPPING d6 "
           f"on d1.TOOL_ID = d6.TOOL_ID "
           f"and d1.RULES_ENABLED=1 and d1.TOOL_ID={tool_id} and d2.WAFER_ID is not null and d2.WAFER_ID <> '' and d2.WAFER_ID <> 'NA' and d4.PRODCAT_ID='Production';")
    print("=================== read_online_uva_data ================")
    print(sql)
    columns = [
        "TOOL_ID",
        "TOOL_NAME",
        "RUN_ID",
        "EQP_NAME",
        "CASE_INFO",
        "PRODUCT_ID",
        "PRODG1",
        "OPER_NO",
        "LOT_ID",
        "WAFER_ID",
        "RECIPE_NAME",
        "INPUT",
        "WINDOW",
        "STATISTICS",
        "parametric_name",
        "START_TIME",
        "PARTITION_ID",
        "STATISTIC_KEY",
        "COLLECTION_KEY",
        "SEQ_ID",
        "PROCESS_TYPE_ID",
        "TIME_STAMP",
        "CALC_TIME_STAMP",
        "TARGET",
        "LOWER_WARNING",
        "UPPER_WARNING",
        "LOWER_CRITICAL",
        "UPPER_CRITICAL",
        "LOWER_OUTLIER",
        "UPPER_OUTLIER",
        "RULES_ENABLED",
        "ALARM_RULE",
        "RESULT",
        "STATUS",
        "REGION",
        "ERROR_MSG",
        "STATISTIC_RESULT",
        "VERSION"
    ]
    return read_as_df(sql, online_engine, columns)


def read_online_uva_data_new_with_tools(case_name, product_id, oper_no, tool_ids, min_start_time, max_start_time):
    sql = (f"select d1.TOOL_ID"
           f",d5.TOOL_NAME"
           f",d1.RUN_ID"
           f",d6.EQP_NAME"
           f",'{case_name}' CASE_INFO"
           f",d2.{product_id} PRODUCT_ID"
           f",d4.PRODG1"
           f",d2.{oper_no} OPER_NO"
           f",d2.LOT_ID"
           f",d2.WAFER_ID"
           f",d2.RECIPE_NAME"
           f",UPPER(d3.INPUT)"
           f",UPPER(d3.WINDOW)"
           f",UPPER(d3.STATISTICS) STATISTICS"
           f",UPPER(CONCAT(d3.INPUT, '#', d3.WINDOW, '#', d3.STATISTICS)) parametric_name"
           f",d1.START_TIME"
           f",d1.PARTITION_ID"
           f",d1.STATISTIC_KEY"
           f",d1.COLLECTION_KEY"
           f",d1.SEQ_ID"
           f",d1.PROCESS_TYPE_ID"
           f",d1.TIME_STAMP"
           f",d1.CALC_TIME_STAMP"
           f",d1.TARGET"
           f",d1.LOWER_WARNING"
           f",d1.UPPER_WARNING"
           f",d1.LOWER_CRITICAL"
           f",d1.UPPER_CRITICAL"
           f",d1.LOWER_OUTLIER"
           f",d1.UPPER_OUTLIER"
           f",d1.RULES_ENABLED"
           f",d1.ALARM_RULE"
           f",d1.RESULT"
           f",d1.STATUS"
           f",d1.REGION"
           f",d1.ERROR_MSG"
           f",d1.STATISTIC_RESULT"
           f",d1.VERSION "
           f"from ODS_E3SUITE.ODS_EES_FD_UVA_HIST partition {case_name_to_doris_partition_name(case_name)} d1 "
           f"join ODS_E3SUITE.ODS_EES_RUN_CONTEXT partition {case_name_to_doris_partition_name(case_name)} d2 "
           f"on d1.TOOL_ID=d2.TOOL_ID and d1.RUN_ID=d2.RUN_ID "
           f"join DWD_DEVELOP.COLLECTION_WITH_CONTEXT d3 "
           f"on d1.COLLECTION_KEY=d3.COLLECTION_KEY and d1.SEQ_ID=d3.SEQ_ID "
           f"join (SELECT DISTINCT PRODUCT_ID, PRODG1, PRODCAT_ID FROM ODS_EDA.ODS_PRODUCT) d4 on d2.{product_id} = d4.PRODUCT_ID "
           f"join ODS_E3RPT.ODS_RPT_TBL_TOOL_INFO d5 on d1.TOOL_ID=d5.TOOL_ID "
           f"join DWD_DEVELOP.TOOL_EQP_MAPPING d6 "
           f"on d1.TOOL_ID = d6.TOOL_ID "
           f"where d1.RULES_ENABLED=1 and d1.TOOL_ID in {tool_ids} and d2.WAFER_ID is not null and d2.WAFER_ID <> '' and d2.WAFER_ID <> 'NA' and d4.PRODCAT_ID='Production';")
    print("=================== read_online_uva_data_new_with_tools ================")
    print(sql)
    columns = [
        "TOOL_ID",
        "TOOL_NAME",
        "RUN_ID",
        "EQP_NAME",
        "CASE_INFO",
        "PRODUCT_ID",
        "PRODG1",
        "OPER_NO",
        "LOT_ID",
        "WAFER_ID",
        "RECIPE_NAME",
        "INPUT",
        "WINDOW",
        "STATISTICS",
        "parametric_name",
        "START_TIME",
        "PARTITION_ID",
        "STATISTIC_KEY",
        "COLLECTION_KEY",
        "SEQ_ID",
        "PROCESS_TYPE_ID",
        "TIME_STAMP",
        "CALC_TIME_STAMP",
        "TARGET",
        "LOWER_WARNING",
        "UPPER_WARNING",
        "LOWER_CRITICAL",
        "UPPER_CRITICAL",
        "LOWER_OUTLIER",
        "UPPER_OUTLIER",
        "RULES_ENABLED",
        "ALARM_RULE",
        "RESULT",
        "STATUS",
        "REGION",
        "ERROR_MSG",
        "STATISTIC_RESULT",
        "VERSION"
    ]
    return read_as_df(sql, online_engine, columns)


def read_online_parametrics():
    sql = (f"select ortti.MODULE"
           f",ortti.SECTIONS"
           f",ortti.TOOL_ID"
           f",ortti.TOOL_NAME"
           f",ortptv.NAME PARAMETRIC_NAME"
           f",ortptv.DATA_TYPE"
           f",ortptv.TABLE_NAME"
           f",ortptv.COLUMN_NAME"
           f",ortetcd.ID EQUIP_TEMPL_CHAMBR_ID"
           f",ortptv.PROC_TYPE_ID"
           f",ortptv.`USAGE`"
           f" from ODS_E3RPT.ODS_RPT_TBL_EQP_TEMPL_CHAMBER_DEF ortetcd"
           f" join ODS_E3SUITE.ODS_EES_TOOL oet"
           f" on oet.EQUIP_TEMPL_CHMBR_ID = ortetcd.ID"
           f" join ODS_E3RPT.ODS_RPT_TBL_PROC_TYPE_VAR ortptv"
           f" on ortetcd.PROC_TYPE_ID = ortptv.PROC_TYPE_ID"
           f" join ODS_E3RPT.ODS_RPT_TBL_TOOL_INFO ortti"
           f" on ortti.TOOL_ID = oet.ID"
           f" where ortptv.TABLE_NAME in('EES_RUN_DATA', 'EES_RUN_CONTEXT');")
    print("=================== read_online_parametrics ================")
    #print(sql)
    columns = [
        "MODULE",
        "SECTIONS",
        "TOOL_ID",
        "TOOL_NAME",
        "PARAMETRIC_NAME",
        "DATA_TYPE",
        "TABLE_NAME",
        "COLUMN_NAME",
        "EQUIP_TEMPL_CHMBR_ID",
        "PROC_TYPE_ID",
        "USAGE"
    ]
    online_engine.dispose()
    return read_as_df(sql, online_engine, columns)


# 由于当前是按照产品来取数，如果直接从etl.DWD_POC_CASE_EES_RUN_DATA选择step，则可能因为数据不全而取不到，所以直接从线上run data选择
def read_online_random_run_id(case_name, tool_id):
    sql = f"select distinct TOOL_ID, RUN_ID from ODS_E3SUITE.ODS_EES_RUN_DATA partition {case_name_to_doris_partition_name(case_name)} where TOOL_ID = {tool_id} limit 1"
    print("=================== read_online_random_run_id ================")
    #print(sql)
    columns = [
        "TOOL_ID",
        "RUN_ID"
    ]
    online_engine.dispose()
    return read_as_df(sql, online_engine, columns)


def read_online_random_run_data(case_name, tool_id, run_id):
    sql = (f"select a.TOOL_ID"
           f",a.PARTITION_ID"
           f",a.TIME_STAMP"
           f",a.START_TIME"
           f",a.RUN_ID"
           f",'{case_name}' CASE_INFO"
           f",a.EVENT_NAME"
           f",a.EVENT_TYPE"
           f",a.ALARM_ID"
           f",a.ALARM_CODE"
           f",a.VAR1,a.VAR2,a.VAR3,a.VAR4,a.VAR5,a.VAR6,a.VAR7,a.VAR8,a.VAR9,a.VAR10"
           f",a.VAR11,a.VAR12,a.VAR13,a.VAR14,a.VAR15,a.VAR16,a.VAR17,a.VAR18,a.VAR19,a.VAR20"
           f",a.VAR21,a.VAR22,a.VAR23,a.VAR24,a.VAR25,a.VAR26,a.VAR27,a.VAR28,a.VAR29,a.VAR30"
           f",a.VAR31,a.VAR32,a.VAR33,a.VAR34,a.VAR35,a.VAR36,a.VAR37,a.VAR38,a.VAR39,a.VAR40"
           f",a.VAR41,a.VAR42,a.VAR43,a.VAR44,a.VAR45,a.VAR46,a.VAR47,a.VAR48,a.VAR49,a.VAR50"
           f",a.VAR51,a.VAR52,a.VAR53,a.VAR54,a.VAR55,a.VAR56,a.VAR57,a.VAR58,a.VAR59,a.VAR60"
           f",a.VAR61,a.VAR62,a.VAR63,a.VAR64,a.VAR65,a.VAR66,a.VAR67,a.VAR68,a.VAR69,a.VAR70"
           f",a.VAR71,a.VAR72,a.VAR73,a.VAR74,a.VAR75,a.VAR76,a.VAR77,a.VAR78,a.VAR79,a.VAR80"
           f",a.VAR81,a.VAR82,a.VAR83,a.VAR84,a.VAR85,a.VAR86,a.VAR87,a.VAR88,a.VAR89,a.VAR90"
           f",a.VAR91,a.VAR92,a.VAR93,a.VAR94,a.VAR95,a.VAR96,a.VAR97,a.VAR98,a.VAR99,a.VAR100"
           f",a.VAR101,a.VAR102,a.VAR103,a.VAR104,a.VAR105,a.VAR106,a.VAR107,a.VAR108,a.VAR109,a.VAR110"
           f",a.VAR111,a.VAR112,a.VAR113,a.VAR114,a.VAR115,a.VAR116,a.VAR117,a.VAR118,a.VAR119,a.VAR120"
           f",a.VAR121,a.VAR122,a.VAR123,a.VAR124,a.VAR125,a.VAR126,a.VAR127,a.VAR128,a.VAR129,a.VAR130"
           f",a.VAR131,a.VAR132,a.VAR133,a.VAR134,a.VAR135,a.VAR136,a.VAR137,a.VAR138,a.VAR139,a.VAR140"
           f",a.VAR141,a.VAR142,a.VAR143,a.VAR144,a.VAR145,a.VAR146,a.VAR147,a.VAR148,a.VAR149,a.VAR150"
           f",a.VAR151,a.VAR152,a.VAR153,a.VAR154,a.VAR155,a.VAR156,a.VAR157,a.VAR158,a.VAR159,a.VAR160"
           f",a.VAR161,a.VAR162,a.VAR163,a.VAR164,a.VAR165,a.VAR166,a.VAR167,a.VAR168,a.VAR169,a.VAR170"
           f",a.VAR171,a.VAR172,a.VAR173,a.VAR174,a.VAR175,a.VAR176,a.VAR177,a.VAR178,a.VAR179,a.VAR180"
           f",a.VAR181,a.VAR182,a.VAR183,a.VAR184,a.VAR185,a.VAR186,a.VAR187,a.VAR188,a.VAR189,a.VAR190"
           f",a.VAR191,a.VAR192,a.VAR193,a.VAR194,a.VAR195,a.VAR196,a.VAR197,a.VAR198,a.VAR199,a.VAR200 "
           f"from ODS_E3SUITE.ODS_EES_RUN_DATA partition {case_name_to_doris_partition_name(case_name)} a "
           f"where a.TOOL_ID ={tool_id} and a.RUN_ID = {run_id} "
           f"order by a.TIME_STAMP")
    print("=================== read_online_random_run_data ================")
    #print(sql)
    columns = [
        "TOOL_ID",
        "PARTITION_ID",
        "TIME_STAMP",
        "START_TIME",
        "RUN_ID",
        "CASE_INFO",
        "EVENT_NAME",
        "EVENT_TYPE",
        "ALARM_ID",
        "ALARM_CODE",
        "VAR1","VAR2","VAR3","VAR4","VAR5","VAR6","VAR7","VAR8","VAR9","VAR10",
        "VAR11","VAR12","VAR13","VAR14","VAR15","VAR16","VAR17","VAR18","VAR19","VAR20",
        "VAR21","VAR22","VAR23","VAR24","VAR25","VAR26","VAR27","VAR28","VAR29","VAR30",
        "VAR31","VAR32","VAR33","VAR34","VAR35","VAR36","VAR37","VAR38","VAR39","VAR40",
        "VAR41","VAR42","VAR43","VAR44","VAR45","VAR46","VAR47","VAR48","VAR49","VAR50",
        "VAR51","VAR52","VAR53","VAR54","VAR55","VAR56","VAR57","VAR58","VAR59","VAR60",
        "VAR61","VAR62","VAR63","VAR64","VAR65","VAR66","VAR67","VAR68","VAR69","VAR70",
        "VAR71","VAR72","VAR73","VAR74","VAR75","VAR76","VAR77","VAR78","VAR79","VAR80",
        "VAR81","VAR82","VAR83","VAR84","VAR85","VAR86","VAR87","VAR88","VAR89","VAR90",
        "VAR91","VAR92","VAR93","VAR94","VAR95","VAR96","VAR97","VAR98","VAR99","VAR100",
        "VAR101","VAR102","VAR103","VAR104","VAR105","VAR106","VAR107","VAR108","VAR109","VAR110",
        "VAR111","VAR112","VAR113","VAR114","VAR115","VAR116","VAR117","VAR118","VAR119","VAR120",
        "VAR121","VAR122","VAR123","VAR124","VAR125","VAR126","VAR127","VAR128","VAR129","VAR130",
        "VAR131","VAR132","VAR133","VAR134","VAR135","VAR136","VAR137","VAR138","VAR139","VAR140",
        "VAR141","VAR142","VAR143","VAR144","VAR145","VAR146","VAR147","VAR148","VAR149","VAR150",
        "VAR151","VAR152","VAR153","VAR154","VAR155","VAR156","VAR157","VAR158","VAR159","VAR160",
        "VAR161","VAR162","VAR163","VAR164","VAR165","VAR166","VAR167","VAR168","VAR169","VAR170",
        "VAR171","VAR172","VAR173","VAR174","VAR175","VAR176","VAR177","VAR178","VAR179","VAR180",
        "VAR181","VAR182","VAR183","VAR184","VAR185","VAR186","VAR187","VAR188","VAR189","VAR190",
        "VAR191","VAR192","VAR193","VAR194","VAR195","VAR196","VAR197","VAR198","VAR199","VAR200"
    ]
    online_engine.dispose()
    return read_as_df(sql, online_engine, columns)


def get_tool_id_step_name(tool_id:int, case_name, conn):
    get_tool_step = f"select step_var, step_name from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_STEP_INFO} where tool_id = '{tool_id}' limit 1"
    records = conn.execute(get_tool_step).fetchall()
    if len(records) == 0:
        try:
            select_step_name, other_step_name = do_get_tool_id_step_name(tool_id, case_name, conn)
            #print(select_step_name, other_step_name)
            tool_id, tool_name, step_var, step_name = list(select_step_name[0])
            sql = f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_STEP_INFO} values({tool_id}, '{tool_name}', '{step_var}', '{step_name}')"
            conn.execute(sql)
            #print(select_step_name, select_step_name[0][2], select_step_name[0][3])
            return (select_step_name[0][2], select_step_name[0][3])
        except Exception as e:
            print(e)
            return ('', '')
    else:
        return records[0]


def do_get_tool_id_step_name(tool_id:int, case_name, conn):
    """
    :param tool_id:
    :return: tuple,
        1. 如果有step信息标记，
            (
                # 选中的step name关系
                [(tool_id, tool_name, step_var, step_name)],


               [
               # 未被选择的step 信息，多条信息，USAGE = 2 的列， 这块暂时不用
               (tool_id, tool_name, step_var, step_name)
                ]
             ),
    2. 如果没有step信息标记
        返回(
            [(tool_id, tool_name, "", "")],
            [],
        )
    """
    # 如果TOOL_ID 不存在结果表
    read_sql = lambda sql:pd.read_sql(sql=sql, con=conn)
    # tool id 映射关系
    # print(tool_id)
    # print(f"SELECT * FROM etl.DWD_POC_TOOL_PARAMETRIC_INFO WHERE TOOL_ID = {tool_id}")
    name_mapping_df = read_sql(sql=f"SELECT * FROM {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_PARAMETRIC_INFO} WHERE TOOL_ID = {tool_id}")
    step_name_mapping_df = name_mapping_df.query("USAGE == 2")

    if len(name_mapping_df) == 0:
        raise ValueError(f"{tool_id} not update to {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_PARAMETRIC_INFO} table")
    else:
        # print(name_mapping_df)
        tool_name = name_mapping_df["TOOL_NAME"].values[0]

    # print(tool_id, tool_name)
    select_step_name = [] # 选择的step_name
    other_step_name = [] # 未被选择的step_name
    select_step_name_null = [(tool_id, tool_name, "", "")]  # 没有step 信息的tool， step 内容为空
    if len(step_name_mapping_df) == 0:
        # 没有step 信息

        # print(f"{tool_id} 没有step 信息")
        return select_step_name_null, []

    else:
        # 有step 信息
        #  随机选一个run id
        tool_first_run_id = read_sql(
            f"select distinct TOOL_ID, RUN_ID from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA} partition {case_name_to_doris_partition_name(case_name)} where TOOL_ID = {tool_id} limit 1")
        #tool_first_run_id = read_online_random_run_id(case_name, tool_id)
        #print(tool_first_run_id)
        # run_id_df = tool_first_run_id["RUN_ID"]
        # todo : 如果run_id 为空，需要做处理
        if len(tool_first_run_id) == 0:
            raise ValueError(f"NO run data for tool_id = {tool_id}")
        run_id = tool_first_run_id["RUN_ID"].values[0]
        tool_run_var = read_sql(
            f"SELECT x.* FROM {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA} partition {case_name_to_doris_partition_name(case_name)} x WHERE TOOL_ID ={tool_id} and RUN_ID = {run_id}")
        #tool_run_var = read_online_random_run_data(case_name, tool_id, run_id)

        # 遍历所有step name， 取他的count
        step_name_unique_tuple_list = []  # 记录step 信息(虚拟名称，真实名称)
        step_name_tuple_unique_count = [] # 记录 step 信息的唯一值个数
        for i, (column_name, real_name) in step_name_mapping_df[["COLUMN_NAME", "PARAMETRIC_NAME"]].iterrows():
            step_unique_cols = [col for col in tool_run_var[column_name].unique().tolist() if col != ""]
            # print(step_unique_cols, tool_run_var[column_name].isna().sum() /len(tool_run_var))
            if len(step_unique_cols) >= 1:
                step_name_unique_tuple_list.append((column_name, real_name))
                step_name_tuple_unique_count.append(len(step_unique_cols))

        # print(step_name_tuple_unique_count)
        if len(step_name_unique_tuple_list) == 0:
            # 有step 标记， 但是，都没有采集数据
            # print(f"{tool_id}有至少一个step标记，但是，唯一值个数为0，为空列")
            return select_step_name_null, []
        else:
            # 有step标记，有采集数据 # 取唯一值多的step name
            argmax = np.array(step_name_tuple_unique_count).argmax()
            # print(
                # f"{tool_id}最多个step 取值的名字是{step_name_unique_tuple_list[argmax]},有{step_name_tuple_unique_count[argmax]}个")

            # if step_name_tuple_unique_count[argmax] >= 1:
                # 分step 处理 , 有多个step

            for i in range(len(step_name_unique_tuple_list)):
                if i == argmax:
                    select_step_name.append(step_name_unique_tuple_list[i])
                else:
                    other_step_name.append(step_name_unique_tuple_list[i])

            return [(tool_id, tool_name, select_step[0], select_step[1].upper().replace("_", "+")) for select_step in  select_step_name], other_step_name
        # step_name = select_step_name[0][1].upper().replace("_", "+").strip()
        # print(f"{tool_id} 选择的step_name 为{step_name}, 未被选择的是{other_step_name}")


def parse_date_with_default(date_str, format_str, default_date):
    try:
        return datetime.datetime.strptime(date_str, format_str)
    except (TypeError, ValueError) as e:
        return default_date


def fix_daterange(date_range_args):
    now = datetime.datetime.now()
    yesterday_now = now - datetime.timedelta(days=1)
    yesterday_now = yesterday_now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date_str = date_range_args.start_date
    end_date_str = date_range_args.end_date

    _start_date = parse_date_with_default(start_date_str, '%Y%m%d', yesterday_now)
    _end_date = parse_date_with_default(end_date_str, '%Y%m%d', yesterday_now) + datetime.timedelta(days=1)
    _end_date = _end_date if _end_date <= now else now.replace(hour=0, minute=0, second=0, microsecond=0)

    return _start_date, _end_date


def case_name_to_doris_partition_name(case_name, namedByMonth=False):
    replaced = f"p{case_name.replace('-', '')}"
    return replaced[0:-2] if namedByMonth else replaced


def clear_context(case_name):
    with local_engine.connect() as conn:
        #conn.execute("truncate table etl.M_DWD_POC_CASE_EES_RUN_DATA;")
        #conn.execute(f"delete from etl.DWD_POC_CASE_FD_UVA_DATA partition {case_name_to_doris_partition_name(case_name, True)} where CASE_INFO='{case_name}';")
        #conn.execute(f"truncate table {LOCAL_SCHEMA_ETL}.{LOCAL_CACHE_TABLE_RUN_DATA_UNPIVOT};")
        conn.execute(f"delete from etl.M_CASE_DISTINCT_TOOL_INFO where CASE_INFO='{case_name}';")
        #conn.execute(f"delete from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_STATS_RESULTS} where CASE_INFO='{case_name}';")


def assemble_products_map(case_name):
    """
    主进程任务
    """
    local_engine.dispose()
    with local_engine.connect() as conn:
        conn.execute(f"truncate table {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_PRODUCTS_MAP};")
        df = read_online_products_map()
        doris_stream_load_from_df(df, local_engine, f"{LOCAL_TABLE_PRODUCTS_MAP}")
        time = datetime.datetime.now()
        conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_ETL_LOG} "
                     f"select '{case_name}', null, null, '{LOCAL_TABLE_PRODUCTS_MAP}', '{time}';")

def assemble_tool_eqp_map(case_name):
    """
    主进程任务
    """
    local_engine.dispose()
    with local_engine.connect() as conn:
        conn.execute(f"truncate table {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_EQP_MAP};")
        df = read_online_tool_eqp_map()
        df['TOOL_ID'] = df['TOOL_ID'].astype(int)
        df['EQP_ID'] = df['EQP_ID'].astype(int)
        doris_stream_load_from_df(df, local_engine, f"{LOCAL_TABLE_TOOL_EQP_MAP}")
        time = datetime.datetime.now()
        conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_ETL_LOG} "
                     f"select '{case_name}', null, null, '{LOCAL_TABLE_TOOL_EQP_MAP}', '{time}';")


@retry(stop_max_attempt_number=STOP_MAX_ATTEMPT_NUMBER, wait_exponential_multiplier=WAIT_EXPONENTIAL_MULTIPLIER)
def get_tools(case_name, min_start_time, max_start_time):
    """
    主进程任务
    """
    get_tool_sql = f"select distinct tool_id, rn from etl.M_CASE_DISTINCT_TOOL_INFO where case_info='{case_name}' order by rn"
    tools = []
    with local_engine.connect() as conn:
        tools = conn.execute(get_tool_sql).fetchall()
        if not tools or len(tools) < 1:
            conn.execute(f"truncate table etl.M_CASE_DISTINCT_TOOL_INFO;")
            df = read_online_distinct_tools(case_name, min_start_time, max_start_time)
            df['tool_id'] = df['tool_id'].astype(int)
            doris_stream_load_from_df(df, local_engine, 'M_CASE_DISTINCT_TOOL_INFO')
            time = datetime.datetime.now()
            conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_ETL_LOG} "
                         f"select '{case_name}', null, null, 'M_CASE_DISTINCT_TOOL_INFO', '{time}'")
            tools = conn.execute(get_tool_sql).fetchall()
    print(f"总共{len(tools)}个腔室的数据")
    return tools


def get_actual_tools(case_name):
    """
    主进程任务
    """
    sql = (f"select TOOL_ID, ROW_NUMBER() over(ORDER BY TOOL_ID) rn "
           f"from ("
           f"select distinct TOOL_ID "
           f"from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA} partition {case_name_to_doris_partition_name(case_name)}"
           f") a;")
    tools = []
    with local_engine.connect() as conn:
        tools = conn.execute(sql).fetchall()
        
    print(f"实际总共{len(tools)}个腔室的数据")
    return tools


def assemble_map(case_name):
    """
    主进程任务
    """
    local_engine.dispose()
    with local_engine.connect() as conn:
        conn.execute(f"truncate table {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_PRODUCT_OPER_MAP};")
        df = read_online_product_oper_map(case_name)
        df['TOOL_ID'] = df['TOOL_ID'].astype(int)
        doris_stream_load_from_df(df, local_engine, f"{LOCAL_TABLE_TOOL_PRODUCT_OPER_MAP}")
        time = datetime.datetime.now()
        conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_ETL_LOG} "
                     f"select '{case_name}', null, null, '{LOCAL_TABLE_TOOL_PRODUCT_OPER_MAP}', '{time}';")


def assemble_product_oper_tools_map():
    local_engine.dispose()
    rs = []
    product_oper_tools_map = dict()
    with local_engine.connect() as conn:
        rs = conn.execute(f"select distinct PRODUCT_ID_COLUMMN, OPER_NO_COLUMMN, TOOL_ID from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_PRODUCT_OPER_MAP};").fetchall()
    for r in rs:
        product_id_column = r[0]
        oper_no_column = r[1]
        tool_id = int(r[2])
        key = ','.join([str(product_id_column), str(oper_no_column)])
        if key not in product_oper_tools_map.keys():
            product_oper_tools_map[key] = []
        product_oper_tools_map[key].append(tool_id)
    return product_oper_tools_map


def load_run_data_with_tools(case_name, tools, min_start_time, max_start_time):
    """
    主进程任务
    """
    # 按照tool_id进行分批etl
    #with local_engine.connect() as conn:
    #    conn.execute(f"delete from etl.DWD_POC_CASE_FD_UVA_DATA partition {case_name_to_doris_partition_name(case_name, True)} where CASE_INFO='{case_name}';")
    #    conn.execute(f"truncate table {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA} partition {case_name_to_doris_partition_name(case_name)};")
    # 按产品
    #product_ids = read_online_products()
    # 全部产品
    product_ids = None
    
    product_oper_tools_map = assemble_product_oper_tools_map()
    #print(product_oper_tools_map)
    #return
    futures = []
    tools_list = [item[0] for item in tools]
    #print(tools_list)
    local_engine.dispose()
    online_engine.dispose()
    for item in product_oper_tools_map.items():
        key = item[0]
        # 只取wafer id不为空和'NA'的run data
        tool_ids = [tool for tool in item[1] if tool in tools_list]
        key_items = key.split(",")
        if len(key_items) > 1:
            product_id_column = key_items[0]
            oper_no_column = key_items[1]
            
            l = len(tool_ids)
            chunksize = 30
            if product_id_column and oper_no_column and l > 0:
                if l >= chunksize:
                    batches = l // chunksize
                    if l % chunksize > 0:
                        batches += 1
                    for i in range(batches):
                        si = i * chunksize
                        ei = min(si + chunksize, l)
                        sub = tool_ids[si:ei]
                        futures.append(executor.submit(do_load_run_data_with_tools, case_name, product_id_column, oper_no_column, sub, min_start_time, max_start_time, product_ids))
                else:
                    futures.append(executor.submit(do_load_run_data_with_tools, case_name, product_id_column, oper_no_column, tool_ids, min_start_time, max_start_time, product_ids))
    print("=================== Waiting load_run_data_with_tools ================")
    if len(futures) > 0:
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
    print("=================== load_run_data_with_tools finished ================")


@retry(stop_max_attempt_number=STOP_MAX_ATTEMPT_NUMBER, wait_exponential_multiplier=WAIT_EXPONENTIAL_MULTIPLIER)
def do_load_run_data_with_tools(case_name, product_id_column, oper_no_column, tool_ids, min_start_time, max_start_time, product_ids):
    """
    多进程任务
    所以这里不建议使用缓存表，比如DWD_POC_CASE_EES_RUN_DATA
    """
    print(f"Processing {product_id_column} {oper_no_column} tools")
    try:
        with local_engine.connect() as conn:
            condition_in = "({})".format(','.join([f"{str(t)}" for t in tool_ids]))
            #只能删除当前进程任务相关的数据，不能直接truncate
            #防止重试时产出脏数据
            #conn.execute(f"delete from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA} partition {case_name_to_doris_partition_name(case_name)} where CASE_INFO='{case_name}' and TOOL_ID in {condition_in};")
            #df = read_online_data_with_tools(case_name, product_id_column, oper_no_column, condition_in, min_start_time, max_start_time, product_ids)
            #
            #if df is not None and len(df) > 0:
            #    df['TOOL_ID'] = df['TOOL_ID'].astype(int)
            #    df['PARTITION_ID'] = df['PARTITION_ID'].astype(int)
            #    df['RUN_ID'] = df['RUN_ID'].astype(int)
            #    
            #    doris_stream_load_from_df(df, local_engine, f"{LOCAL_TABLE_RUN_DATA}", partitions=f'{case_name_to_doris_partition_name(case_name)}')
            #time = datetime.datetime.now()
            #conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_ETL_LOG} "
            #             f"select '{case_name}', {tool_rn}, {tool_id}, '{LOCAL_TABLE_RUN_DATA}', '{time}';")
            
            df = read_online_data_condition_with_tools(case_name, product_id_column, oper_no_column, condition_in, min_start_time, max_start_time, product_ids)
            
            if df is not None and len(df) > 0:
                df['TOOL_ID'] = df['TOOL_ID'].astype(int)
                df['EQP_ID'] = df['EQP_ID'].astype(int)
                
                doris_stream_load_from_df(df, local_engine, f"{LOCAL_TABLE_RUN_DATA_CONDITION}")

            #只能删除当前进程任务相关的数据，不能直接truncate
            #防止重试时产出脏数据
            conn.execute(f"delete from etl.DWD_POC_CASE_FD_UVA_DATA_TEST partition {case_name_to_doris_partition_name(case_name)} where CASE_INFO='{case_name}' and TOOL_ID in {condition_in};")
            df = read_online_uva_data_new_with_tools(case_name, product_id_column, oper_no_column, condition_in, min_start_time, max_start_time)
            if df is not None and len(df) > 0:
                df['TOOL_ID'] = df['TOOL_ID'].astype(int)
                df['PARTITION_ID'] = df['PARTITION_ID'].astype(int)
                df['RUN_ID'] = df['RUN_ID'].astype(int)
                df['SEQ_ID'] = df['SEQ_ID'].astype(int)
                df['RULES_ENABLED'] = df['RULES_ENABLED'].astype(int)
                df['VERSION'] = df['VERSION'].astype(int)
                doris_stream_load_from_df(df, local_engine, 'DWD_POC_CASE_FD_UVA_DATA_TEST', partitions=f'{case_name_to_doris_partition_name(case_name)}')
            #time = datetime.datetime.now()
            #conn.execute(f"insert into etl.DWD_CASE_ETL_LOG "
            #             f"select '{case_name}', {tool_rn}, {tool_id}, 'DWD_POC_CASE_FD_UVA_DATA', '{time}';")
    except Exception as e:
        print(f"Failed to do etl with tool: {e}")
        raise e
    print(f"Processing {product_id_column} {oper_no_column} tools finished")


def load_run_data_with_tool(case_name, tools, min_start_time, max_start_time):
    """
    主进程任务
    """
    # 按照tool_id进行分批etl
    #with local_engine.connect() as conn:
    #    conn.execute(f"truncate table {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA} partition {case_name_to_doris_partition_name(case_name)};")
    # 按产品
    #product_ids = read_online_products()
    # 全部产品
    product_ids = None
    local_engine.dispose()
    online_engine.dispose()
    futures = [executor.submit(do_load_run_data_with_tool, case_name, tl[0], tl[1], min_start_time, max_start_time, product_ids) for tl in tools]
    print("=================== Waiting load_run_data_with_tool ================")
    concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
    print("=================== load_run_data_with_tool finished ================")


@retry(stop_max_attempt_number=STOP_MAX_ATTEMPT_NUMBER, wait_exponential_multiplier=WAIT_EXPONENTIAL_MULTIPLIER)
def do_load_run_data_with_tool(case_name, tool_id, tool_rn, min_start_time, max_start_time, product_ids):
    """
    多进程任务
    所以这里不建议使用缓存表，比如DWD_POC_CASE_EES_RUN_DATA
    """
    print(f"Processing {tool_rn}th tool: {tool_id}")
    product_id = None
    oper_no = None
    
    get_product_oper_sql = (f"select PRODUCT_ID_COLUMMN, OPER_NO_COLUMMN "
                            f"from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_PRODUCT_OPER_MAP} "
                            f"where TOOL_ID={tool_id}")
    try:
        with local_engine.connect() as conn:
            product_opers = conn.execute(get_product_oper_sql).fetchall()
            #print(product_opers)
            for product_oper in product_opers:
                product_id = product_oper[0]
                oper_no = product_oper[1]

            if product_id and oper_no:
                #print(product_id, oper_no)
                #只能删除当前进程任务相关的数据，不能直接truncate
                #防止重试时产出脏数据
                #conn.execute(f"delete from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA} partition {case_name_to_doris_partition_name(case_name)} where CASE_INFO='{case_name}' and TOOL_ID = {tool_id};")
                #df = read_online_data(case_name, product_id, oper_no, tool_id, min_start_time, max_start_time, product_ids)
                #df['PARAMETRIC_NAME'] = df.PARAMETRIC_NAME.str.replace(r'\W', '', regex=True)
                #df_to_doris(df, f"{LOCAL_TABLE_RUN_DATA}",  local_engine, False)
                #df['TOOL_ID'] = df['TOOL_ID'].astype(int)
                #df['PARTITION_ID'] = df['PARTITION_ID'].astype(int)
                #df['RUN_ID'] = df['RUN_ID'].astype(int)
                #doris_stream_load_from_df(df, local_engine, f"{LOCAL_TABLE_RUN_DATA}", partitions=f'{case_name_to_doris_partition_name(case_name)}')
                #time = datetime.datetime.now()
                #conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_ETL_LOG} "
                #             f"select '{case_name}', {tool_rn}, {tool_id}, '{LOCAL_TABLE_RUN_DATA}', '{time}';")

                #只能删除当前进程任务相关的数据，不能直接truncate
                #防止重试时产出脏数据
                conn.execute(f"delete from etl.DWD_POC_CASE_FD_UVA_DATA partition {case_name_to_doris_partition_name(case_name, True)} where CASE_INFO='{case_name}' and TOOL_ID = {tool_id};")
                df = read_online_uva_data_new(case_name, product_id, oper_no, tool_id, min_start_time, max_start_time)
                df['TOOL_ID'] = df['TOOL_ID'].astype(int)
                df['PARTITION_ID'] = df['PARTITION_ID'].astype(int)
                df['RUN_ID'] = df['RUN_ID'].astype(int)
                df['SEQ_ID'] = df['SEQ_ID'].astype(int)
                df['RULES_ENABLED'] = df['RULES_ENABLED'].astype(int)
                df['VERSION'] = df['VERSION'].astype(int)
                doris_stream_load_from_df(df, local_engine, 'DWD_POC_CASE_FD_UVA_DATA', partitions=f'{case_name_to_doris_partition_name(case_name, True)}')
                time = datetime.datetime.now()
                conn.execute(f"insert into etl.DWD_CASE_ETL_LOG "
                             f"select '{case_name}', {tool_rn}, {tool_id}, 'DWD_POC_CASE_FD_UVA_DATA', '{time}';")
    except Exception as e:
        print(f"Failed to do etl with tool: {e}")
        raise e
    print(f"Processing {tool_rn}th tool: {tool_id} finished")


@retry(stop_max_attempt_number=STOP_MAX_ATTEMPT_NUMBER, wait_exponential_multiplier=WAIT_EXPONENTIAL_MULTIPLIER)
def collect_parametrics(case_name):
    """
    主进程任务
    沿用之前使用临时表M_DWD_POC_TOOL_PARAMETRIC_INFO的方案
    """
    local_engine.dispose()
    with local_engine.connect() as conn:
        conn.execute(f"truncate table {LOCAL_SCHEMA_ETL}.{LOCAL_CACHE_TABLE_TOOL_PARAMETRIC_INFO};")
        conn.execute(f"truncate table {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_PARAMETRIC_INFO};")

        df = read_online_parametrics()
        #df['PARAMETRIC_NAME'] = df.PARAMETRIC_NAME.str.replace(r'\W', '', regex=True)
        df['TOOL_ID'] = df['TOOL_ID'].astype(int)
        df['EQUIP_TEMPL_CHMBR_ID'] = df['EQUIP_TEMPL_CHMBR_ID'].astype(int)
        df['PROC_TYPE_ID'] = df['PROC_TYPE_ID'].astype(int)

        doris_stream_load_from_df(df, local_engine, f"{LOCAL_CACHE_TABLE_TOOL_PARAMETRIC_INFO}")
        time = datetime.datetime.now()
        conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_ETL_LOG} "
                     f"select '{case_name}', null, null, '{LOCAL_CACHE_TABLE_TOOL_PARAMETRIC_INFO}', '{time}';")

        conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_PARAMETRIC_INFO} select * from {LOCAL_SCHEMA_ETL}.{LOCAL_CACHE_TABLE_TOOL_PARAMETRIC_INFO};")
        time = datetime.datetime.now()
        conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_ETL_LOG} "
                     f"select '{case_name}', null, null, '{LOCAL_TABLE_TOOL_PARAMETRIC_INFO}', '{time}';")


def do_get_var_names(case_name, tool_id, conn):
    get_var_names_sql = f"SELECT DISTINCT d1.COLUMN_NAME  \
                        from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_PARAMETRIC_INFO} d1  \
                        inner join {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA} partition {case_name_to_doris_partition_name(case_name)} d2 on d1.TOOL_ID = d2.TOOL_ID   \
                        where d1.TOOL_ID={tool_id} and d2.CASE_INFO = '{case_name}' and d1.TABLE_NAME = 'EES_RUN_DATA' and d1.COLUMN_NAME like 'VAR%%';"
    print(get_var_names_sql)
    var_names = conn.execute(get_var_names_sql).fetchall()
    return pd.DataFrame(list(var_names), columns=['VAR_NAME'])


def do_get_var_names_with_tool_ids(case_name, tool_ids, conn):
    condition_in = "({})".format(','.join([f"{str(t)}" for t in tool_ids]))
    get_var_names_sql = f"SELECT DISTINCT d1.COLUMN_NAME  \
                        from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_PARAMETRIC_INFO} d1  \
                        inner join {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA} partition {case_name_to_doris_partition_name(case_name)} d2 on d1.TOOL_ID = d2.TOOL_ID   \
                        where d1.TOOL_ID in {condition_in} and d2.CASE_INFO = '{case_name}' and d1.TABLE_NAME = 'EES_RUN_DATA' and d1.COLUMN_NAME like 'VAR%%';"
    print(get_var_names_sql)
    var_names = conn.execute(get_var_names_sql).fetchall()
    return pd.DataFrame(list(var_names), columns=['VAR_NAME'])


def unpivot(case_name, tools):
    """
    主进程任务
    """
    var_names = pd.DataFrame()
    local_engine.dispose()
    all_futures = []
    step_var_2_tools_map = dict()
    chunksize = 50
    with local_engine.connect() as conn:
        #conn.execute(f"truncate table {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA_UNPIVOT} partition {case_name_to_doris_partition_name(case_name)};")
        for tl in tools:
            tool_id = tl[0]
            tool_id_step_name = get_tool_id_step_name(tool_id, case_name, conn)
            step_var, step_name = list(tool_id_step_name)
            key = f"{step_var}@{step_name}" if step_name else ''
            if key not in step_var_2_tools_map.keys():
                step_var_2_tools_map[key] = []
            step_var_2_tools_map[key].append(tool_id)
        #print(step_var_2_tools_map)
        for item in step_var_2_tools_map.items():
            key_step_var = item[0]
            key_items = key_step_var.split("@")
            key_step_var = key_items[0]
            key_step_name = key_items[1] if len(key_items) > 1 else key_items[0]
            val_tool_ids = item[1]
            var_names = do_get_var_names_with_tool_ids(case_name, val_tool_ids, conn)
            total = len(var_names)
            print(f"Extracted {total} variable names for tools those selected {key_step_var} as their step sensor.")
            l = len(val_tool_ids)
            for i in range(total):
                #if l >= chunksize:
                #        batches = l // chunksize
                #        if l % chunksize > 0:
                #            batches += 1
                #        for b in range(batches):
                #            si = b * chunksize
                #            ei = min(si + chunksize, l)
                #            sub = val_tool_ids[si:ei]
                #            future = executor.submit(do_unpivot_with_tool_ids, case_name, sub, var_names, i, key_step_var, key_step_name)
                #            all_futures.append(future)
                #else:
                #    future = executor.submit(do_unpivot_with_tool_ids, case_name, val_tool_ids, var_names, i, key_step_var, key_step_name)
                #    all_futures.append(future)
                futures = [executor.submit(do_unpivot, case_name, tool_id, var_names, i, key_step_var, key_step_name) for tool_id in val_tool_ids]
                for f in futures:
                    all_futures.append(f)
        
        #for tl in tools:
        #    tool_id = tl[0]
        #    tool_id_step_name = get_tool_id_step_name(tool_id, case_name, conn)
        #    step_var, step_name = list(tool_id_step_name)
        #    print(f"=================== step_var={step_var} step_name={step_name}  ================")
        #    var_names = do_get_var_names(case_name, tool_id, conn)
        #    total = len(var_names)
        #    print(f"Extracted {total} variable names for tool {tool_id}.")
        #    futures = [executor.submit(do_unpivot, case_name, tool_id, var_names, i, step_var, step_name) for i in range(0, total)]
        #    for f in futures:
        #        all_futures.append(f)
    print("=================== Waiting unpivot ================")
    if len(all_futures) > 0:
        concurrent.futures.wait(all_futures, return_when=concurrent.futures.ALL_COMPLETED)
    print("=================== unpivot finished ================")


@retry(stop_max_attempt_number=STOP_MAX_ATTEMPT_NUMBER, wait_exponential_multiplier=WAIT_EXPONENTIAL_MULTIPLIER)
def do_unpivot_with_tool_ids(case_name, tool_ids, var_names, i, step_var, step_name):
    """
    多进程任务
    所以这里不建议使用缓存表M_DWD_POC_CASE_EES_RUN_DATA_UNPIVOT
    """
    condition_in = "({})".format(','.join([f"{str(t)}" for t in tool_ids]))
    assemble_step_var = lambda var_step: f"d1.{var_step} step_value" if var_step is not None and var_step != '' else "'ALL_RUN' step_value"
    try:
        with local_engine.connect() as conn:
            var_name = var_names['VAR_NAME'].iloc[i]
            print(f"Processing {i+1}th var: {var_name} for step {step_var} which named {step_name}")
            #只能删除当前进程任务相关的数据，不能直接truncate
            #防止重试时产出脏数据
            conn.execute(f"delete from {LOCAL_SCHEMA_ETL}.{LOCAL_CACHE_TABLE_RUN_DATA_UNPIVOT} partition {case_name_to_doris_partition_name(case_name)} where CASE_INFO='{case_name}' and TOOL_ID in {condition_in} and sensor_name = '{var_name}' and step_var = '{step_var}';")

            get_run_data_unpivot_sql = (f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_CACHE_TABLE_RUN_DATA_UNPIVOT} "
                                        f"select d1.TOOL_ID"
                                        f",d2.TOOL_NAME"
                                        f",d1.RUN_ID"
                                        f",d1.TIME_STAMP"
                                        f",d1.START_TIME"
                                        f",d1.CASE_INFO"
                                        f",d1.PRODUCT_ID"
                                        f",d1.OPER_NO"
                                        f",d1.LOT_ID"
                                        f",d1.WAFER_ID"
                                        f",d1.RECIPE_NAME"
                                        f",'{var_name}' sensor_name"
                                        f",d2.data_type"
                                        f",d1.{var_name} sensor_value"
                                        f",UPPER(REPLACE(d2.parametric_name, '_', '+')) parametric_name"
                                        f",'{step_var if step_var else ''}' step_var"
                                        f",'{step_name if step_name else ''}' step_name"
                                        f",{assemble_step_var(step_var)}"
                                        f" from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA} partition {case_name_to_doris_partition_name(case_name)} d1"
                                        f" inner join {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_PARAMETRIC_INFO} d2"
                                        f" on d1.TOOL_ID=d2.TOOL_ID"
                                        f" where d1.{var_name} is not null and d1.{var_name} <> '' and d1.CASE_INFO = '{case_name}' and d1.TOOL_ID in {condition_in} and d2.COLUMN_NAME = '{var_name}';")
            print("=================== get_run_data_unpivot ================")
            print(get_run_data_unpivot_sql)
            conn.execute(get_run_data_unpivot_sql)
            time = datetime.datetime.now()
            conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_ETL_LOG} "
                         f"select '{case_name}', null, null, '{var_name}_{LOCAL_CACHE_TABLE_RUN_DATA_UNPIVOT}', '{time}';")
            
            #由于run data有可能出现1秒多次采集的情况，这里保证1秒只取1次采集数据
            get_run_data_unpivot_sql = (f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA_UNPIVOT} "
                                        f"select b.TOOL_ID"
                                        f",b.TOOL_NAME"
                                        f",b.RUN_ID"
                                        f",b.TIME_STAMP"
                                        f",b.START_TIME"
                                        f",b.CASE_INFO"
                                        f",b.PRODUCT_ID"
                                        f",b.OPER_NO"
                                        f",b.LOT_ID"
                                        f",b.WAFER_ID"
                                        f",b.RECIPE_NAME"
                                        f",b.sensor_name"
                                        f",b.data_type"
                                        f",b.sensor_value"
                                        f",b.parametric_name"
                                        f",b.step_var"
                                        f",b.step_name"
                                        f",b.step_value "
                                        f"from("
                                        f"select a.*"
                                        f", row_number() over(partition by a.TOOL_ID,a.RUN_ID,a.sensor_name,a.TIME_STAMP order by a.TIME_STAMP) rn "
                                        f" from {LOCAL_SCHEMA_ETL}.{LOCAL_CACHE_TABLE_RUN_DATA_UNPIVOT} partition {case_name_to_doris_partition_name(case_name)} a"
                                        f" where a.CASE_INFO='{case_name}' and a.TOOL_ID in {condition_in} and a.sensor_name = '{var_name}' and a.step_var = '{step_var}'"
                                        f") b where b.rn=1")
            conn.execute(f"delete from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA_UNPIVOT} partition {case_name_to_doris_partition_name(case_name)} where CASE_INFO='{case_name}' and TOOL_ID in {condition_in} and sensor_name = '{var_name}' and step_var = '{step_var}';")
            print("=================== get_run_data_unpivot ================")
            print(get_run_data_unpivot_sql)
            conn.execute(get_run_data_unpivot_sql)
            time = datetime.datetime.now()
            conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_ETL_LOG} "
                         f"select '{case_name}', null, null, '{var_name}_{LOCAL_TABLE_RUN_DATA_UNPIVOT}', '{time}';")
            #conn.execute(f"delete from {LOCAL_SCHEMA_ETL}.{LOCAL_CACHE_TABLE_RUN_DATA_UNPIVOT} partition {case_name_to_doris_partition_name(case_name)} where CASE_INFO='{case_name}' and TOOL_ID in {condition_in} and sensor_name = '{var_name}' and step_var = '{step_var}';")
            print(f"Processing {i+1}th var: {var_name} for step {step_var} which named {step_name} finished")
    except Exception as e:
        print(f"Failed to do unpivot: {e}")
        raise e


@retry(stop_max_attempt_number=STOP_MAX_ATTEMPT_NUMBER, wait_exponential_multiplier=WAIT_EXPONENTIAL_MULTIPLIER)
def do_unpivot(case_name, tool_id, var_names, i, step_var, step_name):
    """
    多进程任务
    所以这里不建议使用缓存表M_DWD_POC_CASE_EES_RUN_DATA_UNPIVOT
    """
    assemble_step_var = lambda var_step: f"d1.{var_step} step_value" if var_step is not None and var_step != '' else "'ALL_RUN' step_value"
    try:
        with local_engine.connect() as conn:
            var_name = var_names['VAR_NAME'].iloc[i]
            print(f"Processing tool {tool_id} {i+1}th var: {var_name}")
            #只能删除当前进程任务相关的数据，不能直接truncate
            #防止重试时产出脏数据
            conn.execute(f"delete from {LOCAL_SCHEMA_ETL}.{LOCAL_CACHE_TABLE_RUN_DATA_UNPIVOT} partition {case_name_to_doris_partition_name(case_name)} where CASE_INFO='{case_name}' and TOOL_ID={tool_id} and sensor_name = '{var_name}' and step_var = '{step_var}';")

            #get_run_data_unpivot_sql = (f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA_UNPIVOT} "
            #                            f"select b.TOOL_ID"
            #                            f",b.TOOL_NAME"
            #                            f",b.RUN_ID"
            #                            f",date_format(b.TIME_STAMP, '%%Y-%%m-%%d %%H:%%i:%%S') TIME_STAMP"
            #                            f",b.START_TIME"
            #                            f",b.CASE_INFO"
            #                            f",b.PRODUCT_ID"
            #                            f",b.OPER_NO"
            #                            f",b.LOT_ID"
            #                            f",b.WAFER_ID"
            #                            f",b.RECIPE_NAME"
            #                            f",b.sensor_name"
            #                            f",b.data_type"
            #                            f",b.sensor_value"
            #                            f",b.parametric_name"
            #                            f",b.step_var"
            #                            f",b.step_name"
            #                            f",b.step_value "
            #                            f"from("
            #                            f"select a.*"
            #                            f", row_number() over(partition by a.TOOL_ID,a.RUN_ID,a.sensor_name,a.sensor_value,date_format(a.TIME_STAMP, '%%Y-%%m-%%d %%H:%%i:%%S') order by date_format(a.TIME_STAMP, '%%Y-%%m-%%d %%H:%%i:%%S')) rn "
            #                            f"from("
            #                            f"select d1.TOOL_ID"
            #                            f",d2.TOOL_NAME"
            #                            f",d1.RUN_ID"
            #                            f",d1.TIME_STAMP"
            #                            f",d1.START_TIME"
            #                            f",d1.CASE_INFO"
            #                            f",d1.PRODUCT_ID"
            #                            f",d1.OPER_NO"
            #                            f",d1.LOT_ID"
            #                            f",d1.WAFER_ID"
            #                            f",d1.RECIPE_NAME"
            #                            f",'{var_name}' sensor_name"
            #                            f",d2.data_type"
            #                            f",d1.{var_name} sensor_value"
            #                            f",UPPER(REPLACE(d2.parametric_name, '_', '+')) parametric_name"
            #                            f",'{step_var if step_var else ''}' step_var"
            #                            f",'{step_name if step_name else ''}' step_name"
            #                            f",{assemble_step_var(step_var)}"
            #                            f" from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA} partition {case_name_to_doris_partition_name(case_name)} d1"
            #                            f" inner join etl.DWD_POC_TOOL_PARAMETRIC_INFO d2"
            #                            f" on d1.TOOL_ID=d2.TOOL_ID and d2.COLUMN_NAME = '{var_name}'"
            #                            f" where d1.{var_name} is not null and d1.{var_name} <> '' and d1.CASE_INFO = '{case_name}' and d1.TOOL_ID={tool_id}"
            #                            f") a"
            #                            f") b where b.rn=1")
            
            get_run_data_unpivot_sql = (f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_CACHE_TABLE_RUN_DATA_UNPIVOT} "
                                        f"select d1.TOOL_ID"
                                        f",d2.TOOL_NAME"
                                        f",d1.RUN_ID"
                                        f",d1.TIME_STAMP"
                                        f",d1.START_TIME"
                                        f",d1.CASE_INFO"
                                        f",d1.PRODUCT_ID"
                                        f",d1.OPER_NO"
                                        f",d1.LOT_ID"
                                        f",d1.WAFER_ID"
                                        f",d1.RECIPE_NAME"
                                        f",'{var_name}' sensor_name"
                                        f",d2.data_type"
                                        f",d1.{var_name} sensor_value"
                                        f",UPPER(REPLACE(d2.parametric_name, '_', '+')) parametric_name"
                                        f",'{step_var if step_var else ''}' step_var"
                                        f",'{step_name if step_name else ''}' step_name"
                                        f",{assemble_step_var(step_var)}"
                                        f" from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA} partition {case_name_to_doris_partition_name(case_name)} d1"
                                        f" inner join {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_PARAMETRIC_INFO} d2"
                                        f" on d1.TOOL_ID=d2.TOOL_ID and d2.COLUMN_NAME = '{var_name}'"
                                        f" where d1.{var_name} is not null and d1.{var_name} <> '' and d1.CASE_INFO = '{case_name}' and d1.TOOL_ID={tool_id};")
            print("=================== get_run_data_unpivot ================")
            print(get_run_data_unpivot_sql)
            conn.execute(get_run_data_unpivot_sql)
            time = datetime.datetime.now()
            conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_ETL_LOG} "
                         f"select '{case_name}', null, null, '{var_name}_{LOCAL_CACHE_TABLE_RUN_DATA_UNPIVOT}', '{time}';")
            
            #由于run data有可能出现1秒多次采集的情况，这里保证1秒只取1次采集数据
            get_run_data_unpivot_sql = (f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA_UNPIVOT} "
                                        f"select b.TOOL_ID"
                                        f",b.TOOL_NAME"
                                        f",b.RUN_ID"
                                        f",b.TIME_STAMP"
                                        f",b.START_TIME"
                                        f",b.CASE_INFO"
                                        f",b.PRODUCT_ID"
                                        f",b.OPER_NO"
                                        f",b.LOT_ID"
                                        f",b.WAFER_ID"
                                        f",b.RECIPE_NAME"
                                        f",b.sensor_name"
                                        f",b.data_type"
                                        f",b.sensor_value"
                                        f",b.parametric_name"
                                        f",b.step_var"
                                        f",b.step_name"
                                        f",b.step_value "
                                        f"from("
                                        f"select a.*"
                                        f", row_number() over(partition by a.TOOL_ID,a.RUN_ID,a.sensor_name,a.sensor_value,a.TIME_STAMP order by a.TIME_STAMP) rn "
                                        f" from {LOCAL_SCHEMA_ETL}.{LOCAL_CACHE_TABLE_RUN_DATA_UNPIVOT} partition {case_name_to_doris_partition_name(case_name)} a"
                                        f" where a.CASE_INFO='{case_name}' and a.TOOL_ID={tool_id} and a.sensor_name = '{var_name}' and a.step_var = '{step_var}'"
                                        f") b where b.rn=1")
            conn.execute(f"delete from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA_UNPIVOT} partition {case_name_to_doris_partition_name(case_name)} where CASE_INFO='{case_name}' and TOOL_ID={tool_id} and sensor_name = '{var_name}' and step_var = '{step_var}';")
            print("=================== get_run_data_unpivot ================")
            print(get_run_data_unpivot_sql)
            conn.execute(get_run_data_unpivot_sql)
            time = datetime.datetime.now()
            conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_ETL_LOG} "
                         f"select '{case_name}', null, null, '{var_name}_{LOCAL_TABLE_RUN_DATA_UNPIVOT}', '{time}';")
            #conn.execute(f"delete from {LOCAL_SCHEMA_ETL}.{LOCAL_CACHE_TABLE_RUN_DATA_UNPIVOT} partition {case_name_to_doris_partition_name(case_name)} where CASE_INFO='{case_name}' and TOOL_ID={tool_id} and sensor_name = '{var_name}' and step_var = '{step_var}';")
            print(f"Processing tool {tool_id} {i+1}th var: {var_name} finished")
    except Exception as e:
        print(f"Failed to do unpivot: {e}")
        raise e


@retry(stop_max_attempt_number=STOP_MAX_ATTEMPT_NUMBER, wait_exponential_multiplier=WAIT_EXPONENTIAL_MULTIPLIER)
def do_get_oper_no(case_name):
    oper_numbers = pd.DataFrame()
    with local_engine.connect() as conn:
        get_operno_sql = (f"SELECT DISTINCT d1.OPER_NO "
                          f"from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA_UNPIVOT} partition {case_name_to_doris_partition_name(case_name)} d1 "
                          f"where d1.data_type <> 'string' AND d1.data_type is not null "
                          f"AND d1.parametric_name not like '%%STEP%%' AND d1.parametric_name not like '%%COUNT%%' "
                          f"AND d1.RECIPE_NAME is not null "
                          f"AND d1.OPER_NO is not null "
                          f"AND d1.WAFER_ID is not null "
                          f"AND d1.TOOL_NAME is not null "
                          f"AND d1.LOT_ID is not null "
                          f"AND d1.CASE_INFO = '{case_name}' order by d1.OPER_NO DESC;")
        print(get_operno_sql)
        oper_numbers = conn.execute(get_operno_sql).fetchall()
    return pd.DataFrame(list(oper_numbers), columns=['OPER_NO'])


def feature_extraction(case_name):
    """
    主进程任务
    """
    local_engine.dispose()
    oper_numbers = do_get_oper_no(case_name)
    #print(oper_numbers)
    total = len(oper_numbers)
    print(f"Extracting features. Total {total} opers.")
    futures = [executor.submit(do_feature_extraction, case_name, oper_numbers, i) for i in range(0, total)]
    print("=================== Waiting feature_extraction ================")
    concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
    print("=================== feature_extraction finished ================")
    #try:
    #    do_feature_extraction(case_name, oper_no, m, engine)
    #except Exception as e:
    #    do_feature_extraction_batch(case_name, oper_no, m, engine)


@retry(stop_max_attempt_number=STOP_MAX_ATTEMPT_NUMBER, wait_exponential_multiplier=WAIT_EXPONENTIAL_MULTIPLIER)
def do_feature_extraction(case_name, oper_numbers, i):
    """
    多进程任务
    """
    try:
        #print(f"(((((((((((submited {i}")
        with local_engine.connect() as conn:
            time_start = datetime.datetime.now()
            oper_no = oper_numbers['OPER_NO'].iloc[i]
            print(f"Processing {i+1}th oper_no: {oper_no}")
            #只能删除当前进程任务相关的数据，不能直接truncate
            #防止重试时产出脏数据
            #print(f"delete from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_STATS_RESULTS} where CASE_INFO='{case_name}' and OPER_NO = '{oper_no}';")
            conn.execute(f"delete from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_STATS_RESULTS} where CASE_INFO='{case_name}' and OPER_NO = '{oper_no}';")
            #print(f">>>>>>>>>>>>clear {oper_no}")
            feature_extraction_sql = (f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_STATS_RESULTS}"
                                      f"(`WAFER_ID`, `TOOL_ID`, `RUN_ID`, `EQP_NAME`, `PRODUCT_ID`, `PRODG1`, `TOOL_NAME`, `LOT_ID`, `RECIPE_NAME`, `OPER_NO`, `START_TIME`, `CASE_INFO`, `parametric_name`, `mean`, `std`, `min`, `25percentpoint`, `median`, `75percentpoint`, `max`, `range1`) "
                                      f"select d1.`WAFER_ID`"
                                      f",d1.`TOOL_ID`"
                                      f",d1.`RUN_ID`"
                                      f",d3.`EQP_NAME`"
                                      f",d1.`PRODUCT_ID`"
                                      f",d2.`PRODG1`"
                                      f",d1.`TOOL_NAME`"
                                      f",d1.`LOT_ID`"
                                      f",d1.`RECIPE_NAME`"
                                      f",d1.`OPER_NO`"
                                      f",d1.`START_TIME`"
                                      f",'{case_name}' CASE_INFO"
                                      f",CONCAT('STEP_', UPPER(d1.`step_value`), '+', d1.`parametric_name`) as `parametric_name`"
                                      f",d1.`mean`"
                                      f",d1.`std`"
                                      f",d1.`min`"
                                      f",d1.`25percentpoint`"
                                      f",d1.`median`"
                                      f",d1.`75percentpoint`"
                                      f",d1.`max`"
                                      f",d1.`max` - d1.`min` as `range1`"
                                      f" from ("
                                      f" select `WAFER_ID`, `TOOL_ID`, `RUN_ID`, `PRODUCT_ID`, `TOOL_NAME`, `LOT_ID`, `RECIPE_NAME`, `OPER_NO`, `step_value`, `parametric_name`"
                                      f" ,MIN(`START_TIME`) `START_TIME`"
                                      f" ,AVG(`sensor_value`+0) `mean`"
                                      f" ,STDDEV(`sensor_value`+0) `std`"
                                      f" ,PERCENTILE_APPROX(`sensor_value`+0, 0.0) `min`"
                                      f" ,PERCENTILE_APPROX(`sensor_value`+0, 0.25) `25percentpoint`"
                                      f" ,PERCENTILE_APPROX(`sensor_value`+0, 0.5) `median`"
                                      f" ,PERCENTILE_APPROX(`sensor_value`+0, 0.75) `75percentpoint`"
                                      f" ,PERCENTILE_APPROX(`sensor_value`+0, 1) `max`"
                                      f" from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA_UNPIVOT} partition {case_name_to_doris_partition_name(case_name)}"
                                      f" where data_type <> 'string' AND data_type is not null"
                                      f" AND parametric_name not like '%%STEP%%' AND parametric_name not like '%%COUNT%%'"
                                      f" AND RECIPE_NAME is not null"
                                      f" AND WAFER_ID is not null"
                                      f" AND TOOL_NAME is not null"
                                      f" AND LOT_ID is not null"
                                      f" AND OPER_NO = '{oper_no}'"
                                      f" AND CASE_INFO = '{case_name}'"
                                      f" GROUP BY `WAFER_ID`, `TOOL_ID`, `RUN_ID`, `PRODUCT_ID`, `TOOL_NAME`, `LOT_ID`, `RECIPE_NAME`, `OPER_NO`, `step_value`, `parametric_name`"
                                      f") d1"
                                      f" join {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_PRODUCTS_MAP} d2"
                                      f" on d1.`PRODUCT_ID` = d2.`PRODUCT_ID`"
                                      f" join {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_TOOL_EQP_MAP} d3"
                                      f" on d1.`TOOL_ID` = d3.`TOOL_ID`")
            print("=================== do_feature_extraction ================")
            print(feature_extraction_sql)
            conn.execute(feature_extraction_sql)
            time = datetime.datetime.now()
            conn.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_STATS_RESULTS_LOG} "
                         f"select '{oper_no}', {i}, '{time_start}', '{time}', '{time}';")
            print(f"Processing {i+1}th oper_no: {oper_no} finished")
    except Exception as e:
        print(f"Failed to do feature extraction: {e}")
        raise e


@retry(stop_max_attempt_number=STOP_MAX_ATTEMPT_NUMBER, wait_exponential_multiplier=1000)
def get_product_amounts(case_name, oper_no):
    batched_by_product_id_sql = (f"select d1.`PRODUCT_ID`, COUNT(1) AMOUNT"
                              f" from ("
                              f" select `WAFER_ID`, `TOOL_ID`, `RUN_ID`, `PRODUCT_ID`, `TOOL_NAME`, `LOT_ID`, `RECIPE_NAME`, `OPER_NO`, `parametric_name`"
                              f" ,MIN(`START_TIME`) `START_TIME`"
                              f" ,AVG(`sensor_value`+0) `mean`"
                              f" ,STDDEV(`sensor_value`+0) `std`"
                              f" ,PERCENTILE_APPROX(`sensor_value`+0, 0.0) `min`"
                              f" ,PERCENTILE_APPROX(`sensor_value`+0, 0.25) `25percentpoint`"
                              f" ,PERCENTILE_APPROX(`sensor_value`+0, 0.5) `median`"
                              f" ,PERCENTILE_APPROX(`sensor_value`+0, 0.75) `75percentpoint`"
                              f" ,PERCENTILE_APPROX(`sensor_value`+0, 1) `max`"
                              f" from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA_UNPIVOT} partition {case_name_to_doris_partition_name(case_name)}"
                              f" where data_type <> 'string' AND data_type is not null"
                              f" AND parametric_name not like '%%STEP%%' AND parametric_name not like '%%COUNT%%'"
                              f" AND RECIPE_NAME is not null"
                              f" AND WAFER_ID is not null"
                              f" AND TOOL_NAME is not null"
                              f" AND LOT_ID is not null"
                              f" AND OPER_NO = '{oper_no}'"
                              f" AND CASE_INFO = '{case_name}'"
                              f" GROUP BY `WAFER_ID`, `TOOL_ID`, `RUN_ID`, `PRODUCT_ID`, `TOOL_NAME`, `LOT_ID`, `RECIPE_NAME`, `OPER_NO`, `parametric_name`"
                              f") d1 group by d1.`PRODUCT_ID`;")
    print("=================== get_product_amounts ================")
    print(batched_by_product_id_sql)
    mycursor.execute(batched_by_product_id_sql)
    product_amounts = mycursor.fetchall()
    return pd.DataFrame(list(product_amounts), columns=['PRODUCT_ID', 'AMOUNT'])


@retry(stop_max_attempt_number=STOP_MAX_ATTEMPT_NUMBER, wait_exponential_multiplier=1000)
def do_feature_extraction_batch(case_name, oper_no, m):
    time_start = datetime.datetime.now()
    mycursor.execute(f"delete from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_STATS_RESULTS} where CASE_INFO='{case_name}' and OPER_NO = '{oper_no}';")
    df = get_product_amounts(case_name, oper_no)
    total = len(df)
    print("Extracting features batched. Total {total} products for oper_no {oper_no}.")
    for i in range(0, total):
        product_id = df['PRODUCT_ID'].iloc[i]
        feature_extraction_sql = (f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_STATS_RESULTS}"
                                  f"(`WAFER_ID`, `TOOL_ID`, `RUN_ID`, `PRODUCT_ID`, `TOOL_NAME`, `LOT_ID`, `RECIPE_NAME`, `OPER_NO`, `START_TIME`, `CASE_INFO`, `parametric_name`, `mean`, `std`, `min`, `25percentpoint`, `median`, `75percentpoint`, `max`, `range1`) "
                                  f"select d1.`WAFER_ID`"
                                  f",d1.`TOOL_ID`"
                                  f",d1.`RUN_ID`"
                                  f",d1.`PRODUCT_ID`"
                                  f",d1.`TOOL_NAME`"
                                  f",d1.`LOT_ID`"
                                  f",d1.`RECIPE_NAME`"
                                  f",d1.`OPER_NO`"
                                  f",d1.`START_TIME`"
                                  f",'{case_name}' CASE_INFO"
                                  f",d1.`parametric_name`"
                                  f",d1.`mean`"
                                  f",d1.`std`"
                                  f",d1.`min`"
                                  f",d1.`25percentpoint`"
                                  f",d1.`median`"
                                  f",d1.`75percentpoint`"
                                  f",d1.`max`"
                                  f",d1.`max` - d1.`min` as `range1`"
                                  f" from ("
                                  f" select `WAFER_ID`, `TOOL_ID`, `RUN_ID`, `PRODUCT_ID`, `TOOL_NAME`, `LOT_ID`, `RECIPE_NAME`, `OPER_NO`, `parametric_name`"
                                  f" ,MIN(`START_TIME`) `START_TIME`"
                                  f" ,AVG(`sensor_value`+0) `mean`"
                                  f" ,STDDEV(`sensor_value`+0) `std`"
                                  f" ,PERCENTILE_APPROX(`sensor_value`+0, 0.0) `min`"
                                  f" ,PERCENTILE_APPROX(`sensor_value`+0, 0.25) `25percentpoint`"
                                  f" ,PERCENTILE_APPROX(`sensor_value`+0, 0.5) `median`"
                                  f" ,PERCENTILE_APPROX(`sensor_value`+0, 0.75) `75percentpoint`"
                                  f" ,PERCENTILE_APPROX(`sensor_value`+0, 1) `max`"
                                  f" from {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_RUN_DATA_UNPIVOT} partition {case_name_to_doris_partition_name(case_name)}"
                                  f" where data_type <> 'string' AND data_type is not null"
                                  f" AND parametric_name not like '%%STEP%%' AND parametric_name not like '%%COUNT%%'"
                                  f" AND RECIPE_NAME is not null"
                                  f" AND WAFER_ID is not null"
                                  f" AND TOOL_NAME is not null"
                                  f" AND LOT_ID is not null"
                                  f" AND OPER_NO = '{oper_no}'"
                                  f" AND PRODUCT_ID = '{product_id}'"
                                  f" AND CASE_INFO = '{case_name}'"
                                  f" GROUP BY `WAFER_ID`, `TOOL_ID`, `RUN_ID`, `PRODUCT_ID`, `TOOL_NAME`, `LOT_ID`, `RECIPE_NAME`, `OPER_NO`, `parametric_name`"
                                  f") d1;")
        print("=================== do_feature_extraction_batch ================")
        print(feature_extraction_sql)
        mycursor.execute(feature_extraction_sql)
    time = datetime.datetime.now()
    mycursor.execute(f"insert into {LOCAL_SCHEMA_ETL}.{LOCAL_TABLE_STATS_RESULTS_LOG} "
                     f"select '{oper_no}', {m}, '{time_start}', '{time}', '{time}';")


def process(start_date, end_date):
    if end_date > start_date:
        total_seconds = 0
        etl_days = (end_date - start_date).days
        
        start = datetime.datetime.now()
        case_name = start.strftime('%Y-%m-%d')

        assemble_products_map(case_name)
        assemble_tool_eqp_map(case_name)
        start, step_seconds = cal_step_seconds(start)
        print(f">>>>>>>>>>>>> {case_name} assemble_tool_eqp_map takes {step_seconds} seconds.")
        total_seconds += step_seconds
        
        assemble_map(case_name)
        start, step_seconds = cal_step_seconds(start)
        print(f">>>>>>>>>>>>> {case_name} assemble_map takes {step_seconds} seconds.")
        total_seconds += step_seconds
        
        collect_parametrics(case_name)
        start, step_seconds = cal_step_seconds(start)
        print(f">>>>>>>>>>>>> {case_name} collect_parametrics takes {step_seconds} seconds.")
        total_seconds += step_seconds
        
        for i in range(etl_days):
            local_engine.dispose()
            online_engine.dispose()
            today_seconds = 0
            # 本轮etl数据开始日期
            s = start_date + datetime.timedelta(days=i)
            # 每轮etl只处理一天的数据
            e = s + datetime.timedelta(days=1)

            case_name = s.strftime('%Y-%m-%d')
            min_start_time = s.strftime('%Y-%m-%d %H:%M:%S')
            max_start_time = e.strftime('%Y-%m-%d %H:%M:%S')
            print(f'开始处理[{min_start_time}, {max_start_time})的数据, case_name={case_name}')
            clear_context(case_name)
            
            tools = get_tools(case_name, min_start_time, max_start_time)
            start, step_seconds = cal_step_seconds(start)
            print(f">>>>>>>>>>>>> {case_name} get_tools takes {step_seconds} seconds.")
            total_seconds += step_seconds
            
            load_run_data_with_tools(case_name, tools, min_start_time, max_start_time)
            #load_run_data_with_tool(case_name, tools, min_start_time, max_start_time)
            start, step_seconds = cal_step_seconds(start)
            print(f">>>>>>>>>>>>> {case_name} load_run_data_with_tool takes {step_seconds} seconds.")
            today_seconds += step_seconds
            
            #列转行只处理实际取到的tool
            #unpivot(case_name, get_actual_tools(case_name))
            #start, step_seconds = cal_step_seconds(start)
            #print(f">>>>>>>>>>>>> {case_name} do_unpivot takes {step_seconds} seconds.")
            #today_seconds += step_seconds
            #
            #feature_extraction(case_name)
            #start, step_seconds = cal_step_seconds(start)
            #print(f">>>>>>>>>>>>> {case_name} feature_extraction takes {step_seconds} seconds.")
            #today_seconds += step_seconds
            
            print(f"{case_name} takes {today_seconds} seconds.")
            total_seconds += today_seconds
        print(f"Total takes {total_seconds} seconds.")


def cal_step_seconds(start):
    now = datetime.datetime.now()
    return now,(now - start).total_seconds()


# 业务ETL，从Doris到Doris
# 可以通过参数指定要执行ETL的数据日期范围[start_date, end_date]，内部也是一天一天执行的
# end_date为空则默认到昨天，超出昨天会被自动截断到昨天
# 不指定日期范围，默认执行昨天的数据——定时任务跑的时候可不传参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', '-s', type=str, required=False, help='开始日期，包含，格式%Y%m%d')
    parser.add_argument('--end-date', '-e', type=str, required=False, help='结束日期，包含，格式%Y%m%d')
    parser.add_argument('--host', '-H', type=str, default='localhost', help='Doris ip')
    parser.add_argument('--port', '-p', type=int, default=9030, help='Doris port')
    parser.add_argument('--user', '-u', type=str, default='root', help='Doris数据库用户名')
    parser.add_argument('--password', '-P', type=str, default='Nexchip@123', help='Doris数据库密码')
    parser.add_argument('--database', '-d', type=str, default='etl', help='Doris数据库名称')

    args = parser.parse_args()
    start_date, end_date = fix_daterange(args)
    #print(start_date, end_date)
    #exit(0)

    if start_date >= end_date:
        print(f"无效的时间范围：{args.start_date} - {args.end_date}")
    else:
        local_engine = __create_engine(args.host, args.port, args.user, args.password, args.database)
        online_engine = init_online_engine()
        
        @event.listens_for(local_engine, "connect")
        def connect(dbapi_connection, connection_record):
            connection_record.info['pid'] = os.getpid()


        @event.listens_for(local_engine, "engine_connect")
        def connect(conn, branch):
            conn.execute("set delete_without_partition = true")
            conn.execute("set enable_profile = true")
            conn.execute("set enable_cost_based_join_reorder = true")
            #conn.execute("set enable_http_server_v2 = false")
            conn.execute('admin set frontend config("remote_fragment_exec_timeout_ms"="60000");')

        @event.listens_for(local_engine, "checkout")
        def checkout(dbapi_connection, connection_record, connection_proxy):
            pid = os.getpid()
            if connection_record.info['pid'] != pid:
                connection_record.dbapi_connection = connection_proxy.dbapi_connection = None
                raise exc.DisconnectionError(
                    "Connection record belongs to pid %s, "
                    "attempting to check out in pid %s" % (connection_record.info['pid'], pid)
                )
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            process(start_date, end_date)

