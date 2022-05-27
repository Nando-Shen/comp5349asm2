from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import argparse

# init the spark session
spark = SparkSession.builder.appName("COMP5349 A2").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
spark.conf.set("spark.default.parallelism", 40)

parser = argparse.ArgumentParser()
parser.add_argument("--output", help="the output path", default='output.json')
parser.add_argument("--dataset", help="the dataset to be proceeded", default='output.json')
args = parser.parse_args()
output_path = args.output
dataset_opt = args.dataset

data_dic = {'test': 's3://comp5349-2022/test.json',
            'train': 's3://comp5349-2022/train_separate_questions.json',
            'full': 's3://comp5349-2022/CUADv1.json'}

# read in data
test_init_df = spark.read.json(data_dic.get(dataset_opt))

# explode data column
test_data_df = test_init_df.select((explode("data").alias('data')))
# test_data_df.show(5)

# explode paragraphs column
test_paragraphs_df = test_data_df.select((explode("data.paragraphs").alias('paragraphs')),
                                         col("data.title").alias("title"))
# test_paragraphs_df.show(5)

# extract information in the paragraphs columns
test_paragraphs_context_df = test_paragraphs_df.select("title", col("paragraphs.context").alias("context"),
                                                       col("paragraphs.qas").alias("qas")).cache()
# test_paragraphs_context_df.show(5)


# udf function to segment the contract context into 4096 length pieces
@udf(returnType=ArrayType(
    StructType([
        StructField('start', IntegerType(), False),
        StructField('end', IntegerType(), False),
        StructField("source", StringType(), True)
    ])))
def segmentation(row):
    length = len(row)
    i = 0
    j = 4096
    result = []
    while i < length:
        if j > length:
            j = length
        result.append((i, j, row[i:j]))
        i += 2048
        j += 2048
    return result


# explode the segmented context into rows
segmentation_df = test_paragraphs_context_df.select("title", explode(segmentation(col("context"))).alias("context"),
                                                    "qas")
# segmentation_df.show(4)

# explode the segmented qas column
test_seg_paragraphs_df = segmentation_df.select("title", "context", explode("qas").alias("qas"))
# test_seg_paragraphs_df.show(5)

# use explode_outer on answers to ensure the empty answers being retained.
test_paragraphs_qas_df = test_seg_paragraphs_df.select("context", explode_outer("qas.answers").alias("answers"),
                                                       col("qas.id").alias("id"),
                                                       col("qas.is_impossible").alias("is_impossible"),
                                                       col("qas.question").alias("question"), "title")
# test_paragraphs_qas_df.show(5)

# extract the information in the answers column
test_paragraphs_ans_df = test_paragraphs_qas_df.select("context",
                                                       col("answers.answer_start").alias("answer_start"),
                                                       col("answers.text").alias("text"), "id", "is_impossible",
                                                       "question", "title")

# test_paragraphs_ans_df.show(5)


# udf function to label the sequence
@udf(returnType=StructType([
    StructField('answer_start', IntegerType(), False),
    StructField('answer_end', IntegerType(), False),
    StructField("label", StringType(), True)
]))
def label(context, answer_start, text):
    if text is None or answer_start is None:
        return 0, 0, "imp_neg"
    text_length = len(text)
    answer_end = answer_start + text_length
    seq_start, seq_end, seq = context
    if seq_start > answer_end or seq_end < answer_start:
        return 0, 0, "pos_neg"
    elif seq_start < answer_start and seq_end > answer_end:
        return answer_start - seq_start, answer_end - seq_start, "pos"
    elif seq_start > answer_start and seq_end > answer_end:
        return 0, answer_end - seq_start, "pos"
    elif seq_start < answer_start and seq_end < answer_end:
        return answer_start - seq_start, seq_end - seq_start, "pos"
    else:
        return 0, seq_end - seq_start, "pos"


# use udf function to label the generated sequence
test_labeled_ans_df = test_paragraphs_ans_df.withColumn("label",
                                                        label(col("context"), col("answer_start"), col("text")))
# test_labeled_ans_df.show(4)

# extract the information in th nested context and label columns
test_split_ans_df = test_labeled_ans_df.select("context.*", "text", "id", "question", "title", "label.*")
# test_split_ans_df.show(3)

# filter the sample according to their type
test_split_ans_pos = test_split_ans_df.filter(test_split_ans_df.label == "pos").cache()
test_split_ans_imp_neg = test_split_ans_df.filter(test_split_ans_df.label == "imp_neg")
test_split_ans_pos_neg = test_split_ans_df.filter(test_split_ans_df.label == "pos_neg")

# use anti join to remove the repeat sequence (for unique sequence)
test_uni_split_imp_neg = test_split_ans_imp_neg.join(broadcast(test_split_ans_pos), ["source"], "leftanti")
test_uni_split_pos_neg = test_split_ans_pos_neg.join(broadcast(test_split_ans_pos), ["source"], "leftanti")

# group by id on positive sample for the calculation of possible negative
pos_count_for_pos_neg = test_split_ans_pos.groupBy("id").count()
# pos_count_for_pos_neg.show(5)

# group by question on positive sample for the calculation of impossible negative
pos_count_for_imp_neg = test_split_ans_pos.groupBy("question").count()
# pos_count_for_imp_neg.show(5)


# explode the non-segmented qas column
test_paragraphs_inner_df = test_paragraphs_context_df.select("title", "context", explode("qas").alias("qas"))
# test_paragraphs_inner_df.show(5)

# extract the information in the qas column
test_paragraphs_qas_full_df = test_paragraphs_inner_df.select("context", col("qas.answers").alias("answers"),
                                                              col("qas.id").alias("id"),
                                                              col("qas.is_impossible").alias("is_impossible"),
                                                              col("qas.question").alias("question"), "title")
# test_paragraphs_qas_full_df.show(2)

# group by question to calculate the number of false 'is_impossible' question
test_paragraphs_ans_id_full_df = test_paragraphs_qas_full_df.groupBy("question").agg(
    collect_list('is_impossible').alias("is_impossible")) \
    .withColumn("false_count", size(filter(col("is_impossible"), lambda s: s == False)))
# test_paragraphs_ans_id_full_df.show(5)

# join the denominator and numerator columns and calculate the average number for further filtering.
ave_column = when(col("false_count") == 0, 0).otherwise(round(col("count") / col("false_count")))
test_ave_result = test_paragraphs_ans_id_full_df.join(pos_count_for_imp_neg, "question", "left") \
    .withColumn("ave", ave_column).select("question", "ave")
# test_ave_result.show(4)

# join the average number back to the impossible negative samples
test_matched_imp_neg = test_uni_split_imp_neg.join(broadcast(test_ave_result), "question")
# test_matched_imp_neg.show(5)

# join the positive sample number back to the possible negative samples
test_matched_pos_neg = test_uni_split_pos_neg.join(broadcast(pos_count_for_pos_neg), "id")
# test_matched_pos_neg.show(5)

# window function to filter the possible negative samples
pos_window = Window.partitionBy("id").orderBy("answer_start")
pos_neg_result = test_matched_pos_neg.withColumn("row_count", row_number().over(pos_window)) \
    .filter(col("count") >= col("row_count")) \
    .select("source", "question", "answer_start", "answer_end")
# pos_neg_result.show(5)

# window function to filter the impossible negative samples
pos_ave_window = Window.partitionBy("question").orderBy("answer_start")
imp_neg_result = test_matched_imp_neg.withColumn("row_count", row_number().over(pos_ave_window)) \
    .filter(col("ave") >= col("row_count")) \
    .select("source", "question", "answer_start", "answer_end")
# imp_neg_result.show(5)

# select the expected column from the positive samples.
pos_result = test_split_ans_pos.select("source", "question", "answer_start", "answer_end")
# pos_result.show(5)

# union all the samples
result = pos_result.union(imp_neg_result).union(pos_neg_result)
# result.show(5)

# output result and stop the spark server.
result.coalesce(1).write.json(output_path)
spark.stop()
