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


class ParseAlgorithmSqlConfig:
    """
    解析JSON结构的配置信息, 并生成相应的SQL查询语句
    """

    @staticmethod
    def parse_config(json_data: str, properties_config: pd.DataFrame) -> Dict[str, Union[str, Dict[str, str]]]:
        """
        解析 JSON 数据并生成相应的 SQL 查询语句。

        参数:
        json_data (str): 包含配置信息的 JSON 字符串。
        properties_config (pd.DataFrame): 包含属性配置的 DataFrame。

        返回:
        dict: 包含生成的 SQL 查询语句的字典。如果解析失败，返回错误信息字符串。
        """
        try:
            logger.info(f"接收到的 JSON 数据：{json_data}")
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            line, column = e.lineno, e.colno
            logger.error(f"无效的 JSON 数据：在第 {line} 行, 第 {column} 列, 错误信息：{e.msg}")
            logger.error(f"原始 JSON 数据：{json_data}")
            raise ValueError(f"json配置数据解析异常: {e}")

        request_param = data.get("requestParam", {})
        request_id = request_param.get("correlation_filter_upload_id")

        if not request_id:
            logger.error(f"base_sql_query 生成失败, json配置数据中correlation_filter_upload_id属性值异常: {json_data}")
            raise ValueError("base_sql_query 生成失败, json配置数据中correlation_filter_upload_id属性值异常")

        date_range = request_param.get("dateRange", {})
        start_date = date_range.get("start")
        end_date = date_range.get("end")

        # 生成基础 SQL 查询, 区分wafer和site
        algorithm_type = data.get("algorithm")
        if algorithm_type == "correlation_by_wafer":
            base_sql_query = f"SELECT NAME AS WAFER_ID, VAL AS VALUE FROM rca.CONF_ANOMALY WHERE REQUEST_ID='{request_id}'"
        elif algorithm_type == "correlation_by_site":
            base_sql_query = f"SELECT NAME, VAL AS VALUE FROM rca.CONF_ANOMALY WHERE REQUEST_ID='{request_id}'"
        else:
            raise ValueError("base_sql_query 生成失败, json配置数据中algorithm类型异常")

        logging.info("*||*" + "=" * 30 + "-" + f" [base sql query] " + "=" * 30 + "*||*")
        logging.info(f"{base_sql_query}")
        logging.info("*||*" + "=" * 30 + "-" + f" [base sql query] " + "=" * 30 + "*||*")

        uva_select_fields = ["*"]
        inline_select_fields = ["*"]
        wat_select_fields = ["*"]
        process_select_fields = ["*"]

        # 生成其他 SQL 查询
        queries = {
            "uva": ParseAlgorithmSqlConfig.generate_sql_query(json_config=data, algo_type="uva",
                                                              table_name="DWD_FD_UVA_DATA",
                                                              time_field="START_TIME", start_date=start_date,
                                                              end_date=end_date,
                                                              selected_fields=uva_select_fields,
                                                              proper_conf=properties_config),
            "inline": ParseAlgorithmSqlConfig.generate_sql_query(data, "inline",
                                                                 "DWD_INLINE_WAFER_SUMMARY",
                                                                 "MEASURE_TIME", start_date, end_date,
                                                                 selected_fields=inline_select_fields,
                                                                 proper_conf=properties_config),
            "wat": ParseAlgorithmSqlConfig.generate_sql_query(data, "wat",
                                                              "DWD_WAT_WAFER_SUMMARY d1",
                                                              "MEASURE_TIME", start_date,
                                                              end_date,
                                                              selected_fields=wat_select_fields,
                                                              proper_conf=properties_config),
            "process": ParseAlgorithmSqlConfig.generate_sql_query(data, "process",
                                                                  "DWD_WAFER_CONDITION",
                                                                  "START_TIME", start_date,
                                                                  end_date,
                                                                  selected_fields=process_select_fields,
                                                                  proper_conf=properties_config),
        }

        for query_name, sql_query in queries.items():
            if sql_query:
                logging.info("*||*" + "=" * 60 + "-" + f" [{query_name} sql query] " + "=" * 60 + "*||*")
                logger.info(f"{sql_query}")
                logging.info("*||*" + "=" * 60 + "-" + f" [{query_name} sql query] " + "=" * 60 + "*||*")

        # 返回生成的 SQL 查询，仅返回存在的查询
        results = {key: value for key, value in {"base_sql_query": base_sql_query, **queries}.items() if value}

        return results

    @staticmethod
    def generate_sql_query(json_config: dict, algo_type: str, table_name: str, time_field: str, start_date: str,
                           end_date: str,
                           selected_fields: List[str], proper_conf: pd.DataFrame) -> Union[str, None]:
        """
        生成特定表的 SQL 查询语句。

        参数:
        data (dict): 包含查询条件的数据。
        table_name (str): 要查询的表名。
        time_field (str): 时间字段名。
        start_date (str): 查询的开始日期。
        end_date (str): 查询的结束日期。
        selected_fields (list): 要查询的字段列表。
        proper_conf (pd.DataFrame): 包含属性配置的 DataFrame。
        base_sql (str): 基础 SQL 查询语句。

        返回:
        str: 生成的 SQL 查询语句。如果数据为空，返回 None。
        """
        algorithm_type = json_config.get("algorithm")
        data = json_config.get("requestParam", {})
        if not data:
            logger.error(f"json requestParam数据为空, data: {data}")
            return None

        if algo_type not in data.keys():
            return None

        sql_conditions = [f"{time_field} >= '{start_date}' AND {time_field} <= '{end_date}'"]

        # 获取并处理 prodg1 和 productId 的值
        prodg1_values = data.get("prodg1", [])
        productId_values = data.get("productId", [])

        # 处理 prodg1 和 productId 的条件
        for key, values in [("prodg1", prodg1_values), ("productId", productId_values)]:
            if values:
                col_name = column_dicts.get(key)
                value_list = ', '.join([f"'{item}'" for item in values])
                sql_conditions.append(f"{col_name} IN ({value_list})")

        wat_key_mapping_table = f"{proper_conf['doris_db']}.{proper_conf['doris_wat_key_mapping_table_name']}"
        correlation_wafer_table = f"{proper_conf['doris_db']}.{proper_conf['doris_correlation_wafer_table_name']}"

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
                item_type = data.get("wat").get("itemType")
                if item_type:
                    if item_type == 1 or item_type == 2:
                        key_mapping_sql = f" and exists (select 1 from {wat_key_mapping_table} AS km WHERE d1.PARAMETRIC_NAME = km.PARAMETRIC_NAME and km.ITEM_TYPE = {item_type}) "
                        sql_conditions = " WHERE " + " AND ".join(sql_conditions)
                        sql_conditions += key_mapping_sql
                    else:
                        key_mapping_sql = f" and not exists (select 1 from {wat_key_mapping_table} AS km WHERE d1.PARAMETRIC_NAME = km.PARAMETRIC_NAME) "
                        sql_conditions = " WHERE " + " AND ".join(sql_conditions)
                        sql_conditions += key_mapping_sql
                else:
                    sql_conditions = " WHERE " + " AND ".join(sql_conditions)
            elif algo_type in "inline" and algorithm_type == "correlation_by_site":
                sql_conditions = " WHERE DATA_LEVEL='Site' AND " + " AND ".join(sql_conditions)
            else:
                sql_conditions = " WHERE " + " AND ".join(sql_conditions)

        if data.get("correlation_filter_upload_id"):
            upload_id = data.get("correlation_filter_upload_id")
            # in 新表的NAME
            sql_conditions += f" AND WAFER_ID IN (SELECT NAME FROM {correlation_wafer_table} WHERE UPLOAD_ID='{upload_id}')"

            if algo_type in ("inline", "wat") :
                sql_conditions += f" AND RANGE_INDEX = 0 "
                sql_query = f"""
                WITH ranked_summary AS (
                    SELECT {', '.join(selected_fields)} , ROW_NUMBER() OVER (
                                    PARTITION BY `PH_FAB_ID_LABEL`, `PRODG1`, `WAFER_ID`,`OPE_NO`, `PARAMETRIC_NAME` 
                                    ORDER BY MEASURE_TIME DESC
                                ) AS row_num FROM {table_name} {sql_conditions}
                )
                SELECT {', '.join(selected_fields)} 
                FROM ranked_summary
                WHERE row_num = 1
                """
            else:
                sql_query = f"""
                   SELECT {', '.join(selected_fields)} FROM {table_name} {sql_conditions}
                """ 
        else:
            logger.error(
                f"source_sql_query 生成失败, json配置数据中correlation_filter_upload_id属性值为空: {json_config}")
            raise ValueError("source_sql_query 生成失败, json配置数据中correlation_filter_upload_id属性值为空")
            
        return sql_query


if __name__ == '__main__':
    # 示例 JSON 数据
    jsonData = '''
{
  "algorithm": "correlation_by_wafer",
  "requestId": "269",
  "requestParam": {
    "anomaly": {
      "analysis_type": "fdc",
      "chamber": "",
      "data_type": "",
      "eqp": "",
      "ope_no": "",
      "parameter": ""
    },
    "dateRange": {
      "end": "2024-03-06 19:50:49",
      "start": "2021-12-06 19:50:49"
    },
    "flagMergeAllProdg1": "0",
    "flagMergeAllProductId": "0",
    "inline": {
      "mergeOperno": [],
      "operNo": [
        "OPEN.IKAS01",
        "OPEN.IKAS02"
      ]
    },
    "process": {
      "flagMergeAllChamber": "0",
      "operNo": [
        "OPEN.IKAS01",
        "OPEN.IKAS02"
      ]
    },
    "prodg1": [],
    "productId": [],
    "uva": {
      "eqp": [
        "IKASEQP02"
      ],
      "flagMergeAllChamber": "0",
      "mergeChamber": [],
      "mergeOperno": [],
      "operNo": [
        "OPEN.IKAS01",
        "OPEN.IKAS02"
      ],
      "tool": [
        "IKASEQP02_A"
      ]
    },
    "wat": {
      "itemType": 1,
      "watItem": []
    }
  }
}
    '''
    # 获取配置值
    # host = config.get('database', 'host')
    # port = config.get('database', 'port')
    # name = config.get('database', 'name')

    # 执行解析函数并打印结果
    print(json.dumps(ParseAlgorithmSqlConfig.parse_config(jsonData, properties_df)))
