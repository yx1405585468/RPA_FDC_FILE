from pyspark.sql import SparkSession, DataFrame


def read(spark_session: SparkSession, query_sql: str, properties_config) -> DataFrame:
    url = properties_config["doris_jdbc_url"]
    properties = {
        "user": properties_config["doris_user"],
        "password": properties_config["doris_password"]
    }
    # 使用 SQL 查询的结果创建 DataFrame
    result_df = spark_session.read.jdbc(url=url, table=f"({query_sql}) as subquery", properties=properties)
    print(f"执行查询SQL：{query_sql}")
    print(f"查询到{result_df.count()}条数据")
    return result_df
