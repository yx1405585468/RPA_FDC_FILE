# -*- coding: utf-8 -*-
# time: 2024/6/20 15:36
# file: parse_json_to_sql.py
# author: waylee
import json
import logging

# 配置日志记录
from typing import Union, List, Dict, Iterable

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)
# 配置彩色日志
# coloredlogs.install(
#     level=logging.INFO,
#     fmt='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
#     logger=logger
# )
#


column_dicts = {
    "prodg1": "PRODG1",
    "productId": "PRODUCT_ID",
    "operNo": "OPE_NO",
    "eqp": "EQP_NAME",
    "tool": "CHAMBER_NAME",
    "recipeName": "RECIPE_NAME",
    "parametricName": "PARAMETRIC_NAME",
}


class ParseSqlQueryConfigTest:
    """
    解析JSON结构的配置信息, 并生成相应的SQL查询语句
    """

    @staticmethod
    def parse_json_to_sql(json_data: str) -> Dict[str, Union[str, Dict[str, str]]]:
        """
        解析 JSON 数据并生成相应的 SQL 查询语句。

        参数:
        json_data (str): 包含配置信息的 JSON 字符串。

        返回:
        dict: 包含生成的 SQL 查询语句的字典。如果解析失败，返回错误信息字符串。
        """
        try:
            logger.info(f"接收到的 JSON 数据：{json_data}")
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            line, column = e.lineno, e.colno
            logger.error(f"无效的 JSON 数据：在第 {line} 行，第 {column} 列。错误信息：{e.msg}")
            logger.error(f"原始 JSON 数据：{json_data}")
            raise ValueError(f"json数据解析异常: {e}")

        request_param = data.get("requestParam", {})
        analysis_type = request_param.get("anomaly").get("analysis_type")
        data_type = request_param.get("anomaly", {}).get("dataType")

        anomaly_params = {
            "upload_id": request_param.get("uploadId"),
            "ope_no": request_param.get("anomaly", {}).get("ope_no"),
            "eqp": request_param.get("anomaly", {}).get("eqp"),
            "chamber": request_param.get("anomaly", {}).get("chamber"),
            "parameter": request_param.get("anomaly", {}).get("parameter"),
            "layer_id": request_param.get("anomaly", {}).get("layer_id")
        }
        date_range = request_param.get("dateRange", {})
        start_date = date_range.get("start")
        end_date = date_range.get("end")

        # value_col = "MIN_VAL" if data_type == "MIN" else "MAX_VAL"

        # 生成基础 SQL 查询
        base_sql_query = ParseSqlQueryConfigTest.generate_base_sql_query(analysis_type.lower(), anomaly_params, data_type,
                                                                         start_date,
                                                                         end_date).strip().rstrip()
        if not base_sql_query:
            logger.error(f"base_sql_query 生成失败,json数据结构异常: {json_data}")
            raise ValueError("base_sql_query 生成失败, json数据结构异常")

        logger.info(f"生成 base_sql_query：{base_sql_query}")

        uva_select_fields = ["*"]
        inline_select_fields = ["*"]
        wat_select_fields = ["*"]

        # 生成其他 SQL 查询
        queries = {
            "uva": ParseSqlQueryConfigTest.generate_sql_query(data=request_param, algo_type="uva",
                                                              table_name="DWD_FD_UVA_DATA",
                                                              time_field="START_TIME", start_date=start_date,
                                                              end_date=end_date,
                                                              selected_fields=uva_select_fields),
            "inline": ParseSqlQueryConfigTest.generate_sql_query(request_param, "inline",
                                                             "DWD_INLINE_WAFER_SUMMARY",
                                                             "MEASURE_TIME", start_date, end_date,
                                                                 selected_fields=inline_select_fields),
            "wat": ParseSqlQueryConfigTest.generate_sql_query(request_param, "wat",
                                                          "DWD_WAT_WAFER_SUMMARY d1",
                                                          "MEASURE_TIME", start_date,
                                                              end_date,
                                                              selected_fields=wat_select_fields)
        }

        for query_name, sql_query in queries.items():
            if sql_query:
                logger.info(f"生成{query_name}: {sql_query}")

        # 返回生成的 SQL 查询，仅返回存在的查询
        results = {key: value for key, value in {"base_sql_query": base_sql_query, **queries}.items() if value}

        return results

    @staticmethod
    def generate_base_sql_query(analysis_type: str, anomaly_params: dict, data_type: str,
                                start_date: str, end_date: str) -> str:
        """
        根据分析类型和异常参数生成基础 SQL 查询语句。

        参数:
        analysis_type (str): 分析类型，如 'fdc', 'inline', 'wat', 'cp', 'defect' 等。
        anomaly_params (dict): 包含异常参数的字典。
        data_type (str): 数据类型，如 'MIN' 或 'MAX'。
        start_date (str): 查询的开始日期。
        end_date (str): 查询的结束日期。

        返回:
        str: 生成的 SQL 查询语句。如果未匹配到指定类型，返回错误信息。
        """
        valid_analysis_types = ['fdc', 'inline', 'wat', 'cp', 'defect']
        if analysis_type not in valid_analysis_types:
            raise ValueError(
                f"analysis_type 错误：未知的分析类型 '{analysis_type}'。请使用以下类型之一：{', '.join(valid_analysis_types)}")

        sql_queries = {
            'fdc': f"""
                SELECT DISTINCT WAFER_ID, `RESULT` AS `value` 
                FROM DWD_FD_UVA_DATA 
                WHERE START_TIME >= '{start_date}' AND START_TIME <= '{end_date}'
                AND OPE_NO = '{anomaly_params["ope_no"]}' AND EQP_NAME = '{anomaly_params["eqp"]}' 
                AND CHAMBER_NAME = '{anomaly_params["chamber"]}' AND PARAMETRIC_NAME = '{anomaly_params["parameter"]}' 
                AND `RESULT` IS NOT NULL AND WAFER_ID IN (SELECT NAME FROM rca.CONF_WAFER WHERE UPLOAD_ID = '{anomaly_params["upload_id"]}')""",
            'inline': f"""
                SELECT DISTINCT WAFER_ID, {data_type} AS `value` 
                FROM DWD_INLINE_WAFER_SUMMARY 
                WHERE MEASURE_TIME >= '{start_date}' AND MEASURE_TIME <= '{end_date}'
                AND OPE_NO = '{anomaly_params["ope_no"]}' AND PARAMETRIC_NAME = '{anomaly_params["parameter"]}' 
                AND {data_type} IS NOT NULL AND WAFER_ID IN (SELECT NAME FROM rca.CONF_WAFER WHERE UPLOAD_ID = '{anomaly_params["upload_id"]}')""",
            'wat': f"""
                SELECT DISTINCT WAFER_ID, {data_type} AS `value` 
                FROM DWD_WAT_WAFER_SUMMARY 
                WHERE MEASURE_TIME >= '{start_date}' AND MEASURE_TIME <= '{end_date}'
                AND PARAMETRIC_NAME = '{anomaly_params["parameter"]}' 
                AND {data_type} IS NOT NULL AND WAFER_ID IN (SELECT NAME FROM rca.CONF_WAFER WHERE UPLOAD_ID = '{anomaly_params["upload_id"]}')""",
            'cp': f"""
                SELECT DISTINCT WAFER_ID, YIELD AS `value` 
                FROM DWD_CP_WAFER_SUMMARY 
                WHERE MEASURE_TIME >= '{start_date}' AND MEASURE_TIME <= '{end_date}'
                AND YIELD IS NOT NULL AND WAFER_ID IN (SELECT NAME FROM rca.CONF_WAFER WHERE UPLOAD_ID = '{anomaly_params["upload_id"]}')""",
            'defect': f"""
                SELECT DISTINCT WAFER_ID, {anomaly_params["parameter"]} AS `value` 
                FROM DWD_DEFECT_WAFER_SUMMARY 
                WHERE INSPECTION_TIME >= '{start_date}' AND INSPECTION_TIME <= '{end_date}'
                AND LAYER_ID = '{anomaly_params["layer_id"]}' 
                AND {anomaly_params["parameter"]} IS NOT NULL AND WAFER_ID IN (SELECT NAME FROM rca.CONF_WAFER WHERE UPLOAD_ID = '{anomaly_params["upload_id"]}')"""
        }

        formatted_sql_queries = {
            analysis_type: sql_queries[analysis_type].replace('\n', ' ').replace('\t', '').replace('  ', ' ')
            for analysis_type in sql_queries
        }

        return formatted_sql_queries.get(analysis_type, "")

    @staticmethod
    def generate_sql_query(data: dict, algo_type: str, table_name: str, time_field: str, start_date: str,
                           end_date: str, selected_fields: List[str]) -> Union[
        str, None]:
        """
        生成特定表的 SQL 查询语句。

        参数:
        data (dict): 包含查询条件的数据。
        table_name (str): 要查询的表名。
        time_field (str): 时间字段名。
        start_date (str): 查询的开始日期。
        end_date (str): 查询的结束日期。
        selected_fields (list): 要查询的字段列表。

        返回:
        str: 生成的 SQL 查询语句。如果数据为空，抛出异常。
        """
        if not data:
            logger.error(f"json requestParam数据为空, data: {data}")
            raise ValueError("json requestParam数据为空")

        if algo_type not in data.keys():
            return None

        sql_conditions = [f"{time_field} >= '{start_date}' AND {time_field} <= '{end_date}'"]
        sql_query = f"SELECT {', '.join(selected_fields)} FROM {table_name}"

        # 获取并处理 prodg1 和 productId 的值
        prodg1_values = data.get("prodg1", [])
        productId_values = data.get("productId", [])

        # 处理 prodg1 和 productId 的条件
        for key, values in [("prodg1", prodg1_values), ("productId", productId_values)]:
            if values:
                col_name = column_dicts.get(key)
                value_list = ', '.join([f"'{item}'" for item in values])
                sql_conditions.append(f"{col_name} IN ({value_list})")

        # if item_type is not None:
        #     if item_type == 1 or item_type == 2:
        #         key_mapping_sql = f" and exists (select 1 from {wat_key_mapping_table_name} AS km WHERE d1.PARAMETRIC_NAME = km.PARAMETRIC_NAME and km.ITEM_TYPE = {item_type}) "
        #     else:
        #         key_mapping_sql = f" and not exists (select 1 from {wat_key_mapping_table_name} AS km WHERE d1.PARAMETRIC_NAME = km.PARAMETRIC_NAME) "

        # 处理其他字段的条件
        for k, _ in data.items():
            if k in algo_type:
                for k2, v in data.get(k).items():
                    if k2 != "itemType" and k2 in column_dicts:
                        colum_name = column_dicts.get(k2)
                        if isinstance(v, Iterable) and not isinstance(v, str) and v:
                            items_list = ', '.join([f"'{item}'" for item in v])
                            sql_conditions.append(f"{colum_name} IN ({items_list})")
                        elif isinstance(v, (int, str)) and v:
                            sql_conditions.append(f"{colum_name} = '{v}'")

        if sql_conditions:
            if algo_type in "wat":
                if data.get("wat").get("itemType"):
                    item_type = data.get("wat").get("itemType")
                    if item_type == 1 or item_type == 2:
                        key_mapping_sql = f" and exists (select 1 from wat_key_mapping_table_name AS km WHERE d1.PARAMETRIC_NAME = km.PARAMETRIC_NAME and km.ITEM_TYPE = 1) "
                        sql_query += " WHERE " + " AND ".join(sql_conditions)
                        sql_query += key_mapping_sql
                    else:
                        key_mapping_sql = f" and not exists (select 1 from wat_key_mapping_table_name AS km WHERE d1.PARAMETRIC_NAME = km.PARAMETRIC_NAME) "
                        sql_query += " WHERE " + " AND ".join(sql_conditions)
                        sql_query += key_mapping_sql


            else:
                sql_query += " WHERE " + " AND ".join(sql_conditions)

        return sql_query


if __name__ == '__main__':
    import configparser

    # 示例 JSON 数据
    jsonData = '''
    {
    "requestId": "693",
    "algorithm": "correlation_by_wafer",
    "requestParam": {
        "dateRange": {
            "start": "2021-03-18 10:07:09",
            "end": "2024-06-18 10:07:09"
        },
        "anomaly": {
            "analysis_type": "FDC",
            "ope_no": "OPEN.IKAS01",
            "eqp": "IKASEQP03",
            "layer_id": "",
            "chamber": "IKASEQP03_A",
            "parameter": "RFFORWARDPOWER_TOP#FLOW_01#RANGE",
            "data_type": ""
        },
        "uploadId": "7d8217f70e6a4f2c86e0dbcb328a5244",
        "prodg1": [],
        "productId": [],
        "flagMergeAllProdg1": "0",
        "flagMergeAllProductId": "0",
        "uva": {
            "operNo": ["OPEN.IKAS01"],
            "eqp": ["IKASEQP02"],
            "tool": [ "IKASEQP02_A" ],
            "recipeName": [],
            "flagMergeAllChamber": "0",
            "mergeProdg1": [],
            "mergeProductId": [],
            "mergeEqp": [],
            "mergeChamber": [],
            "mergeOperno": [],
            "mergeLayerId": []
        }
    }
}
    '''
    # 获取配置值
    # host = config.get('database', 'host')
    # port = config.get('database', 'port')
    # name = config.get('database', 'name')

    # 执行解析函数并打印结果
    print(json.dumps(ParseSqlQueryConfigTest.parse_json_to_sql(jsonData)))
