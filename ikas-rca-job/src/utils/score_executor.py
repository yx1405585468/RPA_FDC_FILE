from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col,coalesce,lit, monotonically_increasing_id, row_number
)
from pyspark.sql.window import Window
def get_score_df(spark_session: SparkSession, properties_config,  data_frame: DataFrame, analysis_type: str = "CORRELATION" ) -> DataFrame:
    parametric_name_score_query = 'SELECT * FROM PARAMETRIC_NAME_SCORE'
    # 读取Doris数据库中的数据
    parametric_name_score_df = spark_session.read.format("jdbc") \
        .option("url", properties_config['doris_jdbc_url']) \
        .option("user", properties_config['doris_user']) \
        .option("password", properties_config['doris_password']) \
        .option("driver", "com.mysql.cj.jdbc.Driver") \
        .option("query", parametric_name_score_query) \
        .option("numPartitions", "3") \
        .option("fetchsize", "1000") \
        .load()
     # 重命名 parametric_name_score_df 中的 PARAMETRIC_NAME 和 ope_no 列
    parametric_name_score_df = parametric_name_score_df.withColumnRenamed("PARAMETRIC_NAME", "PARAMETRIC_NAME_SCORE") \
                                                       .withColumnRenamed("OPE_NO", "OPE_NO_SCORE") \
                                                       .withColumnRenamed("EQP_NAME", "EQP_NAME_SCORE")
    # 将 PARAMETRIC_NAME_SCORE 表的数据与 finally_res_df 进行左连接
    # 创建 join 条件
    join_conditions = [
        col("data_frame.PARAMETRIC_NAME") == col("parametric_name_score_df.PARAMETRIC_NAME_SCORE"),
        (
            (col("data_frame.EQP_NAME").isNull() & col("parametric_name_score_df.EQP_NAME_SCORE").isNull()) |
            (col("data_frame.EQP_NAME") == col("parametric_name_score_df.EQP_NAME_SCORE"))
        ),
        (
            (col("data_frame.OPE_NO").isNull() & col("parametric_name_score_df.OPE_NO_SCORE").isNull()) |
            (col("data_frame.OPE_NO") == col("parametric_name_score_df.OPE_NO_SCORE"))
        )
    ]
    
    # 将 PARAMETRIC_NAME_SCORE 表的数据与 data_frame 进行左连接
      # 将 PARAMETRIC_NAME_SCORE 表的数据与 data_frame 进行左连接
    joined_df = data_frame.alias("data_frame").join(
        parametric_name_score_df.alias("parametric_name_score_df"),
        join_conditions,
        how="left"
    )

    # 根据 WEIGHT 列进行排序
    if analysis_type == "CORRELATION":
        # 计算 importance 列
        joined_df = joined_df.withColumn("WEIGHT", col("WEIGHT") * coalesce(col("SCORE"), lit(1)))
        sorted_df = joined_df.orderBy(col("WEIGHT").desc(), col("COUNT").desc())
    else:
        # 计算 importance 列
        joined_df = joined_df.withColumn("WEIGHT_PERCENT", col("WEIGHT_PERCENT") * coalesce(col("SCORE"), lit(1)))
        sorted_df = joined_df.orderBy(col("WEIGHT_PERCENT").desc())
    
    # 重新赋值 INDEX_NO 列
    window_spec = Window.orderBy(monotonically_increasing_id())
    sorted_df = sorted_df.withColumn("INDEX_NO", row_number().over(window_spec))
    return sorted_df