import json
from functools import reduce

import pyspark
import pandas as pd
from pyspark.sql.functions import lit, col, sum as spark_sum

from src.exceptions.rca_base_exception import RCABaseException
from src.utils import read_jdbc_executor
from src.defect.defect_algorithm import ExertDefectAlgorithm
from src.inline.inline_bysite_algorithm import ExertInlineBySite
from src.inline.inline_bywafer_algorithm import ExertInlineByWafer
from src.uva.uva_algorithm import ExertUvaAlgorithm
from src.wat.wat_bysite_algorithm import ExertWatBySite
from src.wat.wat_bywafer_algorithm import ExertWatByWafer

from src.uva.build_query import build_uva_query
from src.inline.build_query import build_inline_query
from src.wat.build_query import build_wat_query
from src.defect.build_query import build_defect_query

from src.uva.uva_main import parse_JSON_config as uva_config
from src.inline.inline_main import parse_JSON_config as inline_config
from src.wat.wat_main import parse_JSON_config as wat_config
from src.defect.defect_main import parse_JSON_config as defect_config


class HybridAlgorithms:

    def __init__(self, df_info: pd.DataFrame, sparkSession, properties_config) -> pyspark.sql.dataframe:
        self.df_info = df_info
        self.sparkSession = sparkSession
        self.properties_config = properties_config

    @staticmethod
    def uva(df_info: pd.DataFrame, sparkSession, properties_config):
        parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber = uva_config(
            df_info)
        query_sql = build_uva_query(parse_dict, properties_config)
        doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)
        final_res_uva = ExertUvaAlgorithm.fit_uva_model(df=doris_spark_df,
                                                        grpby_list=grpby_list,
                                                        request_id=request_id,
                                                        merge_operno_list=merge_operno,
                                                        merge_prodg1_list=merge_prodg1,
                                                        merge_product_list=merge_product,
                                                        merge_eqp_list=merge_eqp,
                                                        merge_chamber_list=merge_chamber)
        return final_res_uva

    @staticmethod
    def inline_by_wafer(df_info: pd.DataFrame, sparkSession, properties_config):
        parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, good_site, bad_site = inline_config(
            df_info)
        query_sql = build_inline_query(parse_dict, properties_config)
        doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)

        final_res_inline_by_wafer = ExertInlineByWafer.fit_by_wafer_model(df=doris_spark_df,
                                                                          request_id=request_id,
                                                                          grpby_list=grpby_list,
                                                                          merge_operno_list=merge_operno,
                                                                          merge_prodg1_list=merge_prodg1,
                                                                          merge_product_list=merge_product)
        return final_res_inline_by_wafer

    @staticmethod
    def inline_by_site(df_info: pd.DataFrame, sparkSession, properties_config):
        parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, good_site, bad_site = inline_config(
            df_info)
        query_sql = build_inline_query(parse_dict, properties_config)
        doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)

        final_res_inline_by_site = ExertInlineBySite.fit_by_site_model(df=doris_spark_df,
                                                                       request_id=request_id,
                                                                       merge_operno_list=merge_operno,
                                                                       merge_prodg1_list=merge_prodg1,
                                                                       merge_product_list=merge_product,
                                                                       grpby_list=grpby_list,
                                                                       good_site_columns=good_site,
                                                                       bad_site_columns=bad_site)
        return final_res_inline_by_site

    @staticmethod
    def wat_by_wafer(df_info: pd.DataFrame, sparkSession, properties_config):
        parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, good_site, bad_site = wat_config(
            df_info)
        query_sql = build_wat_query(parse_dict, properties_config)
        doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)

        final_res_wat_by_wafer = ExertWatByWafer.fit_by_wafer_model(df=doris_spark_df,
                                                                    request_id=request_id,
                                                                    grpby_list=grpby_list,
                                                                    merge_operno_list=merge_operno,
                                                                    merge_prodg1_list=merge_prodg1,
                                                                    merge_product_list=merge_product)
        return final_res_wat_by_wafer

    @staticmethod
    def wat_by_site(df_info: pd.DataFrame, sparkSession, properties_config):
        parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, good_site, bad_site = wat_config(
            df_info)
        query_sql = build_wat_query(parse_dict, properties_config)
        doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)
        final_res_wat_by_site = ExertWatBySite.fit_by_site_model(df=doris_spark_df,
                                                                 request_id=request_id,
                                                                 grpby_list=grpby_list,
                                                                 merge_operno_list=merge_operno,
                                                                 merge_prodg1_list=merge_prodg1,
                                                                 merge_product_list=merge_product,
                                                                 good_site_columns=good_site,
                                                                 bad_site_columns=bad_site)
        return final_res_wat_by_site

    @staticmethod
    def defect(df_info: pd.DataFrame, sparkSession, properties_config):
        parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, merge_layerid, good_site, bad_site = defect_config(
            df_info)
        query_sql = build_defect_query(parse_dict, properties_config)
        doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)
        final_res_defect = ExertDefectAlgorithm.fit_defect_model(df=doris_spark_df,
                                                                 request_id=request_id,
                                                                 merge_layerid_list=merge_layerid,
                                                                 merge_prodg1_list=merge_prodg1,
                                                                 merge_product_list=merge_product,
                                                                 grpby_list=grpby_list)
        return final_res_defect


class ExertHybridAlgorithms:
    def __init__(self, config, sparkSession, properties_config, keywords=None):
        if keywords is None:
            keywords = ['uva', 'inline_by_wafer', 'inline_by_site', 'inline', 'wat',
                        'wat_by_wafer', 'wat_by_site', 'defect']
        self.config = config
        self.keywords = keywords
        self.sparkSession = sparkSession
        self.properties_config = properties_config

    @staticmethod
    def extract_and_combine(config, keywords):
        base_params = {k: v for k, v in config['requestParam'].items() if k not in keywords}
        print("base_params:", base_params)
        combined_dict = {}

        for key in keywords:
            if key in config['requestParam']:
                new_dict = {**base_params, **config['requestParam'][key]}
                combined_dict[key] = new_dict
        return combined_dict

    def run(self):
        combined_dicts = self.extract_and_combine(self.config, self.keywords)
        print("combined_dicts:", "\n", combined_dicts)
        keys = combined_dicts.keys()
        print(keys)

        results = []

        if 'uva' in keys:
            df_info = pd.DataFrame({"requestId": [self.config["requestId"]],
                                    "requestParam": combined_dicts['uva']})

            final_res_uva = HybridAlgorithms.uva(df_info, self.sparkSession, self.properties_config)
            final_res_uva = final_res_uva.withColumn('ANALYSIS_NAME', lit('UVA'))
            results.append(final_res_uva)

        if 'inline_by_wafer' in keys or 'inline' in keys:
            df_info = pd.DataFrame({"requestId": [self.config["requestId"]],
                                    "requestParam": combined_dicts['inline_by_wafer']})
            final_res_inline_by_wafer = HybridAlgorithms.inline_by_wafer(df_info, self.sparkSession, self.properties_config)
            final_res_inline_by_wafer = final_res_inline_by_wafer.withColumn('ANALYSIS_NAME', lit('INLINE_BY_WAFER'))
            results.append(final_res_inline_by_wafer)

        if 'inline_by_site' in keys:
            df_info = pd.DataFrame({"requestId": [self.config["requestId"]],
                                    "requestParam": combined_dicts['inline_by_site']})
            final_res_inline_by_site = HybridAlgorithms.inline_by_wafer(df_info, self.sparkSession, self.properties_config)
            final_res_inline_by_site = final_res_inline_by_site.withColumn('ANALYSIS_NAME', lit('INLINE_BY_SITE'))
            results.append(final_res_inline_by_site)

        if 'wat_by_wafer' in keys or 'wat' in keys:
            df_info = pd.DataFrame({"requestId": [self.config["requestId"]],
                                    "requestParam": combined_dicts['wat_by_wafer']})
            final_res_wat_by_wafer = HybridAlgorithms.wat_by_wafer(df_info, self.sparkSession, self.properties_config)
            final_res_wat_by_wafer = final_res_wat_by_wafer.withColumn('ANALYSIS_NAME', lit('WAT_BY_WAFER'))
            results.append(final_res_wat_by_wafer)

        if 'wat_by_site' in keys:
            df_info = pd.DataFrame({"requestId": [self.config["requestId"]],
                                    "requestParam": combined_dicts['wat_by_site']})
            final_res_wat_by_site = HybridAlgorithms.wat_by_wafer(df_info, self.sparkSession, self.properties_config)
            final_res_wat_by_site = final_res_wat_by_site.withColumn('ANALYSIS_NAME', lit('WAT_BY_SITE'))
            results.append(final_res_wat_by_site)

        if 'defect' in keys:
            df_info = pd.DataFrame({"requestId": [self.config["requestId"]],
                                    "requestParam": combined_dicts['defect']})
            final_res_defect = HybridAlgorithms.defect(df_info, self.sparkSession, self.properties_config)
            final_res_defect = final_res_defect.withColumn('ANALYSIS_NAME', lit('DEFECT'))
            results.append(final_res_defect)

        if results:
            final_result = reduce(lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True), results)
            total_importance = final_result.select(spark_sum("WEIGHT")).collect()[0][0]
            final_result = final_result.withColumn("WEIGHT", col("WEIGHT") / total_importance)
            final_result = final_result.orderBy(col("WEIGHT").desc())
            return final_result
        else:
            msg = "未指定具体算法."
            raise RCABaseException(msg)


if __name__ == '__main__':
    import os
    # from pyspark.sql import SparkSession
    #
    # os.environ['PYSPARK_PYTHON'] = '/usr/local/python-3.9.13/bin/python3'
    # spark = SparkSession.builder \
    #     .appName("pandas_udf_by_site") \
    #     .config('spark.sql.session.timeZone', 'Asia/Shanghai') \
    #     .config("spark.scheduler.mode", "FAIR") \
    #     .config('spark.driver.memory', '8g') \
    #     .config('spark.driver.cores', '12') \
    #     .config('spark.executor.memory', '8g') \
    #     .config('spark.executor.cores', '12') \
    #     .config('spark.cores.max', '12') \
    #     .config('spark.driver.host', '192.168.22.28') \
    #     .master("spark://192.168.12.47:7077,192.168.12.48:7077") \
    #     .getOrCreate()

    json_config_ = {"requestId": "269",
                    "algorithm": "hybid_by_wafer",  # hybid_by_site
                    "requestParam": {"dateRange": {"start": "2021-12-06 19:50:49",
                                                   "end": "2024-03-06 19:50:49"},
                                     "prodg1": [],
                                     "productId": [],
                                     "flagMergeAllProdg1": "0",
                                     "flagMergeAllProductId": "0",
                                     "uploadId": "07c0b16fc5b0438f98cda7762fcd9514",
                                     'waferId': {'good': [],
                                                 'bad': []},
                                     "uva": {
                                         "operNo": ["OPEN.IKAS01", "OPEN.IKAS02"],
                                         "eqp": ["IKASEQP02"],
                                         "tool": ["IKASEQP02_A"],
                                         "recipeName": [],
                                         "mergeEqp": [],
                                         "mergeChamber": [],
                                         "mergeOperno": []
                                     },
                                     "inline": {
                                         "operNo": ["OPEN.IKAS01", "OPEN.IKAS02"],
                                         "mergeOperno": []
                                     },
                                     "wat": {
                                         "itemType": 1,
                                         "watItem": []
                                     }
                                     }
                    }

    df_ = pd.DataFrame({"requestId": [json_config_["requestId"]],
                        "requestParam": [json.dumps(json_config_["requestParam"])]})

    request_id_ = df_["requestId"].values[0]
    request_params = df_["requestParam"].values[0]
    print(type(request_params))

    parse_dict_ = json.loads(request_params)
    print(parse_dict_)
    print(type(parse_dict_))

    final_results = ExertHybridAlgorithms(config=json_config_, sparkSession=None, properties_config=None).run()
