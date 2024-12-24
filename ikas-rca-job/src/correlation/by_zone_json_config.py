# -*- coding: utf-8 -*-
# time: 2024/7/11 11:55
# file: by_zone_json_config.py
# author: waylee

import pymysql
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


class ByZoneConfigPare:
    """
    用于从MySQL数据库查询数据并转换为JSON格式的工具类。
    """

    def __init__(self, host, port, user, password, database, charset='utf8mb4'):
        """
        初始化数据库连接参数。

        :param host: 数据库主机地址
        :param port: 数据库端口号
        :param user: 数据库用户名
        :param password: 数据库密码
        :param database: 数据库名称
        :param charset: 数据库字符集，默认为'utf8mb4'
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.conn = None
        self.cursor = None

    def connect(self):
        """
        连接到MySQL数据库。
        """
        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset
            )
            self.cursor = self.conn.cursor()
            logging.info("Successfully connected to the database.")
        except pymysql.Error as e:
            logging.error(f"Error connecting to the database: {e}")
            raise e

    def close(self):
        """
        关闭数据库连接。
        """
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logging.info("Database connection closed.")

    def query_to_json(self, query_sql):
        """
        执行SQL查询并将结果转换为JSON格式。

        :param query_sql: 要执行的SQL查询语句
        :return: 查询结果的字典类型
        """
        try:
            self.cursor.execute(query_sql)
            results = self.cursor.fetchall()
            logging.info(f"Query executed successfully: {query_sql}")
        except pymysql.Error as e:
            logging.error(f"Error executing query: {e}")
            raise e

        data = {
            "mode_info": {}
        }

        for row in results:
            name, mode, sites, analysis, label = row
            sites_array = sites.split(',')
            if analysis not in data["mode_info"]:
                data["mode_info"][analysis] = {}
            if label not in data["mode_info"][analysis]:
                data["mode_info"][analysis][label] = {}
            data["mode_info"][analysis][label][name] = sites_array

        return data

    def run(self, query_sql):
        """
        执行整个流程：连接数据库、执行查询、转换为JSON格式并关闭连接。

        :param query_sql: 要执行的SQL查询语句
        :return: 查询结果的字典类型
        """
        self.connect()
        jsonData = self.query_to_json(query_sql)
        self.close()
        return jsonData


# 使用示例
if __name__ == "__main__":
    db_config = {
        'host': '192.168.13.17',
        'port': 9030,
        'user': 'root',
        'password': '123456',
        'database': 'rca'
    }

    # query = "SELECT NAME, MODE, SITES, ANALYSIS, LABEL FROM rca.CONF_MODE"
    query = "SELECT NAME, MODE, SITES, ANALYSIS, LABEL FROM rca.CONF_MODE WHERE ANALYSIS = 'INLINE'"

    json_config = {"requestId": "269",
                    "algorithm": "inline_by_wafer",
                    "requestParam": {"dateRange": {"start": "2021-12-06 19:50:49",
                                                   "end": "2024-03-06 19:50:49"},
                                     "operNo": ["1U.CDG10", "1U.CDG20", "1V.PQA10", "2U.PQA10", "2V.PQW10", "3U.PQA10",
                                                "6V.CDG10", "7U.PQA10",
                                                "7U.PQX10", "TM.PQX10", "XX.PQW01", "XX.PQX02", "1U.EQW20", "1U.PQW10",
                                                "1U.PQX10", "1V.PQX10",
                                                "1V.PQX20", "2U.PQW10", "2U.PQX10"],
                                     "uploadId": "84f6a2b46a5443ec9797347424402058",
                                     "flagMergeAllProdg1": "0",
                                     "flagMergeAllProductId": "0",
                                     "flagMergeAllChamber": "0",
                                     "mergeProdg1": [],
                                     "mergeProductId": [],
                                     "mergeEqp": [],
                                     "mergeChamber": [],
                                     "mergeOperno": [{
                                         "1U.CDG10,1U.CDG20,1V.PQA10,2U.PQA10,2V.PQW10,3U.PQA10,6V.CDG10,7U.PQA10,7U.PQX10,TM.PQX10,XX.PQW01,XX.PQX02,1U.EQW20,1U.PQW10,1U.PQX10,1V.PQX10,1V.PQX20,2U.PQW10,2U.PQX10":
                                             ["1U.CDG10", "1U.CDG20", "1V.PQA10", "2U.PQA10",
                                              "2V.PQW10", "3U.PQA10", "6V.CDG10", "7U.PQA10", "7U.PQX10",
                                              "TM.PQX10", "XX.PQW01", "XX.PQX02", "1U.EQW20", "1U.PQW10",
                                              "1U.PQX10", "1V.PQX10", "1V.PQX20", "2U.PQW10", "2U.PQX10"]}],
                                     "goodSite": ["SITE4_VAL", "SITE8_VAL", "SITE9_VAL", "SITE12_VAL", "SITE13_VAL"],
                                     "badSite": ["SITE2_VAL", "SITE6_VAL", "SITE7_VAL", "SITE10_VAL", "SITE11_VAL"]
                                     }
                    }

    mysql_to_json = ByZoneConfigPare(**db_config)
    mode_info_json = mysql_to_json.run(query)
    json_config["requestParam"]["mode_info"]=mode_info_json["mode_info"]
    print(json.dumps(json_config))
