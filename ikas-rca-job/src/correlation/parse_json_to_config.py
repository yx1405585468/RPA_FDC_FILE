# -*- coding: utf-8 -*-
# time: 2024/6/20 15:32
# file: parse_json_to_config.py
# author: waylee
import json
import logging
import json
# 配置日志记录
from typing import Dict, Union, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

logger = logging.getLogger(__name__)


class ParseAlgorithmJsonConfig:
    """
    解析JSON结构的配置信息, 并生成相应的算法配置信息
    """

    @staticmethod
    def parse_config(json_data: str) -> Dict[str, any]:
        """
        解析主配置函数，处理输入的JSON数据，提取并解析各个部分的配置信息

        :param json_data: 输入的JSON字符串
        :return: 解析后的配置信息字典
        """
        try:
            data = json.loads(json_data)
            request_id = data.get("requestId")
            algorithm = data.get("algorithm")
            results = {
                "request_id": request_id,
                "algorithm": algorithm,
                "uva": {},
                "wat": {},
                "inline": {},
                "process": {}
            }

            # 解析uva配置
            if "uva" in data.get("requestParam", {}):
                uva_config = data["requestParam"]["uva"]
                results["uva"] = ParseAlgorithmJsonConfig.parse_uva_config(uva_config, data)

            # 解析wat配置
            if "wat" in data.get("requestParam", {}):
                wat_config = data["requestParam"]["wat"]
                results["wat"] = ParseAlgorithmJsonConfig.parse_wat_config(wat_config, data)

            # 解析inline配置
            if "inline" in data.get("requestParam", {}):
                inline_config = data["requestParam"]["inline"]
                results["inline"] = ParseAlgorithmJsonConfig.parse_inline_config(inline_config, data)

            # 解析inline配置
            if "process" in data.get("requestParam", {}):
                process_config = data["requestParam"]["process"]
                results["process"] = ParseAlgorithmJsonConfig.parse_process_config(process_config, data)
            return results

        except json.JSONDecodeError as e:
            line, column = e.lineno, e.colno
            logger.error(f"无效的 JSON 数据：在第 {line} 行，第 {column} 列。错误信息：{e.msg}")
            logger.error(f"原始 JSON 数据：{json_data}")
            return f"无效的 JSON 数据：在第 {line} 行，第 {column} 列。错误信息：{e.msg}"

    @staticmethod
    def get_merge_list(config: Dict[str, any], key: str) -> Optional[List[str]]:
        """
        从配置中获取合并列表

        :param config: 配置字典
        :param key: 要获取的键名
        :return: 对应的值列表，如果键不存在则返回None
        """
        try:
            return list(config.get(key)) if config.get(key) else None
        except KeyError:
            return None

    @staticmethod
    def parse_uva_config(config: Dict[str, any], data: Dict[str, any]) -> Dict[str, any]:
        """
        解析uva配置

        :param config: uva配置字典
        :param data: 完整的数据字典
        :return: 解析后的uva配置信息字典
        """
        group_by_list, merge_prodg1, merge_product, merge_eqp, merge_chamber, merge_operno = ParseAlgorithmJsonConfig.get_merge_list(
            config, 'groupByList'), ParseAlgorithmJsonConfig.get_merge_list(config,
                                                                            'mergeProdg1'), ParseAlgorithmJsonConfig.get_merge_list(
            config, 'mergeProductId'), ParseAlgorithmJsonConfig.get_merge_list(config,
                                                                               'mergeEqp'), ParseAlgorithmJsonConfig.get_merge_list(
            config, 'mergeChamber'), ParseAlgorithmJsonConfig.get_merge_list(config,
                                                                             'mergeOperno')
        if group_by_list is None or len(group_by_list) == 0:
            group_by_list = ["PRODG1", "PRODUCT_ID", "OPE_NO", "EQP_NAME", "CHAMBER_NAME"]
            flag_merge_prodg1 = data.get("requestParam").get('flagMergeAllProdg1')
            flag_merge_product_id = data.get("requestParam").get('flagMergeAllProductId')
            flag_merge_chamber = config.get('flagMergeAllChamber')

            if flag_merge_prodg1 == '1':
                merge_prodg1 = None
                merge_product = None
                group_by_list = ['OPE_NO', "EQP_NAME", 'CHAMBER_NAME']
                if flag_merge_chamber == '1':
                    group_by_list = ['OPE_NO', "EQP_NAME"]
            elif flag_merge_product_id == '1':
                merge_product = None
                group_by_list = ["PRODG1", "OPE_NO", "EQP_NAME", "CHAMBER_NAME"]
                if flag_merge_chamber == '1':
                    group_by_list = ["PRODG1", 'OPE_NO', "EQP_NAME"]
            elif flag_merge_chamber == '1':
                merge_chamber = None
                group_by_list = ["PRODG1", "PRODUCT_ID", "OPE_NO", "EQP_NAME"]

        return {
            "group_by_list": group_by_list,
            "merge_prodg1_list": merge_prodg1,
            "merge_product_list": merge_product,
            "merge_eqp_list": merge_eqp,
            "merge_chamber_list": merge_chamber,
            "merge_operno_list": merge_operno
        }

    @staticmethod
    def parse_wat_config(config: Dict[str, any], data: Dict[str, any]) -> Dict[str, any]:
        """
        解析wat配置

        :param config: wat配置字典
        :param data: 完整的数据字典
        :return: 解析后的wat配置信息字典
        """
        group_by_list, merge_prodg1, merge_product, merge_eqp, merge_chamber, merge_operno = ParseAlgorithmJsonConfig.get_merge_list(
            config, 'groupByList'), ParseAlgorithmJsonConfig.get_merge_list(config,
                                                                            'mergeProdg1'), ParseAlgorithmJsonConfig.get_merge_list(
            config, 'mergeProductId'), ParseAlgorithmJsonConfig.get_merge_list(config,
                                                                               'mergeEqp'), ParseAlgorithmJsonConfig.get_merge_list(
            config, 'mergeChamber'), ParseAlgorithmJsonConfig.get_merge_list(config,
                                                                             'mergeOperno')
        if group_by_list is None or len(group_by_list) == 0:
            group_by_list = ["PRODG1", "PRODUCT_ID"]
            flag_merge_prodg1 = data.get("requestParam").get('flagMergeAllProdg1')
            flag_merge_product_id = data.get("requestParam").get('flagMergeAllProductId')

            if flag_merge_prodg1 == '1':
                merge_prodg1 = None
                merge_product = None
                group_by_list = []
            elif flag_merge_product_id == '1':
                merge_product = None
                group_by_list = ["PRODG1"]

        return {
            "group_by_list": group_by_list,
            "merge_prodg1_list": merge_prodg1,
            "merge_product_list": merge_product,
            "merge_eqp_list": merge_eqp,
            "merge_chamber_list": merge_chamber,
            "merge_operno_list": merge_operno
        }

    @staticmethod
    def parse_inline_config(config: Dict[str, any], data: Dict[str, any]) -> Dict[str, any]:
        """
        解析inline配置

        :param config: inline配置字典
        :param data: 完整的数据字典
        :return: 解析后的inline配置信息字典
        """
        group_by_list, merge_prodg1, merge_product, merge_eqp, merge_chamber, merge_operno = ParseAlgorithmJsonConfig.get_merge_list(
            config, 'groupByList'), ParseAlgorithmJsonConfig.get_merge_list(config,
                                                                            'mergeProdg1'), ParseAlgorithmJsonConfig.get_merge_list(
            config, 'mergeProductId'), ParseAlgorithmJsonConfig.get_merge_list(config,
                                                                               'mergeEqp'), ParseAlgorithmJsonConfig.get_merge_list(
            config, 'mergeChamber'), ParseAlgorithmJsonConfig.get_merge_list(config,
                                                                             'mergeOperno')
        if group_by_list is None or len(group_by_list) == 0:
            group_by_list = ["PRODG1", "PRODUCT_ID", "OPE_NO"]
            flag_merge_prodg1 = data.get("requestParam").get('flagMergeAllProdg1')
            flag_merge_product_id = data.get("requestParam").get('flagMergeAllProductId')

            if flag_merge_prodg1 == '1':
                merge_prodg1 = None
                merge_product = None
                group_by_list = ['OPE_NO']
            elif flag_merge_product_id == '1':
                merge_product = None
                group_by_list = ["PRODG1", "OPE_NO"]

        return {
            "group_by_list": group_by_list,
            "merge_prodg1_list": merge_prodg1,
            "merge_product_list": merge_product,
            "merge_eqp_list": merge_eqp,
            "merge_chamber_list": merge_chamber,
            "merge_operno_list": merge_operno
        }

    @staticmethod
    def parse_process_config(config: Dict[str, any], data: Dict[str, any]) -> Dict[str, any]:
        """
        解析process配置

        :param config: wat配置字典
        :param data: 完整的数据字典
        :return: 解析后的wat配置信息字典
        """
        group_by_list, merge_prodg1, merge_product, merge_eqp, merge_chamber, merge_operno = ParseAlgorithmJsonConfig.get_merge_list(
            config, 'groupByList'), ParseAlgorithmJsonConfig.get_merge_list(config,
                                                                            'mergeProdg1'), ParseAlgorithmJsonConfig.get_merge_list(
            config, 'mergeProductId'), ParseAlgorithmJsonConfig.get_merge_list(config,
                                                                               'mergeEqp'), ParseAlgorithmJsonConfig.get_merge_list(
            config, 'mergeChamber'), ParseAlgorithmJsonConfig.get_merge_list(config,
                                                                             'mergeOperno')
        if group_by_list is None or len(group_by_list) == 0:
            group_by_list = ["PRODG1", "PRODUCT_ID", "EQP_NAME", "CHAMBER_NAME"]
            flag_merge_prodg1 = data.get("requestParam").get('flagMergeAllProdg1')
            flag_merge_product_id = data.get("requestParam").get('flagMergeAllProductId')
            flag_merge_chamber = config.get('flagMergeAllChamber')

            if flag_merge_prodg1 == '1':
                merge_prodg1 = None
                merge_product = None
                group_by_list = ["EQP_NAME", 'CHAMBER_NAME']
                if flag_merge_chamber == '1':
                    group_by_list = [ "EQP_NAME"]
            elif flag_merge_product_id == '1':
                merge_product = None
                group_by_list = ["PRODG1", "EQP_NAME", "CHAMBER_NAME"]
                if flag_merge_chamber == '1':
                    group_by_list = ["PRODG1", "EQP_NAME"]
            elif flag_merge_chamber == '1':
                merge_chamber = None
                group_by_list = ["PRODG1", "PRODUCT_ID", "EQP_NAME"]

        return {
            "group_by_list": group_by_list,
            "merge_prodg1_list": merge_prodg1,
            "merge_product_list": merge_product,
            "merge_eqp_list": merge_eqp,
            "merge_chamber_list": merge_chamber,
            "merge_operno_list": merge_operno
        }


if __name__ == '__main__':
    # 示例 JSON 数据
    jsonData = '''
        {
            "requestId": "269",
            "algorithm": "correlation_by_wafer",
            "requestParam": {
                "dateRange": {
                    "start": "2021-12-06 19:50:49",
                    "end": "2024-03-06 19:50:49"
                },
                "anomaly": {
                    "analysis_type": "UVA",
                    "ope_no": "dasd",
                    "eqp": "11",
                    "layer_id": "",
                    "chamber": "dad",
                    "parameter": "11",
                    "data_type": "MIN"
                },
                "upload_id": "",
                "prodg1": ["IKAS.PROJECT1","IKAS.PROJECT2"],
                "productId": ["IKAS.PROJECT.0A05","IKAS.PROJECT.0A06"],
                "flagMergeAllProdg1": "0",
                "flagMergeAllProductId": "0",
                "uva": {
                    "operNo": ["OPEN.IKAS01","OPEN.IKAS02"],
                    "eqp": ["IKASEQP02"],
                    "tool": ["IKASEQP02_A"],
                    "recipeName": ["IKAS.RECIPE.01"],
                    "flagMergeAllChamber": "0",
                    "mergeProdg1": [],
                    "mergeProductId": [],
                    "mergeOperno": [],
                    "mergeEqp": [],
                    "mergeChamber": []
                },
                "inline": {
                    "operNo": ["OPEN.IKAS01","OPEN.IKAS02"],
                    "mergeOperno": ["IKASEQP02_A","IKASEQP01_A"]
                },
                "wat": {
                    "itemType": 2,
                    "parametricName": ["parametric_1","parametric_2"]
                }
            }
        }
    '''
    print(json.dumps(ParseAlgorithmJsonConfig.parse_config(jsonData)))
    # print(ParseJsonConfig.parse_all_configs(jsonData))
