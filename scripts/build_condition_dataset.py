import argparse
import json
from pathlib import Path

import pandas as pd



def load_csv(path):
    """
    读取manifest文件
    """

    return pd.read_csv(path)



def merge_data(
        phase1,
        phase2
):
    """
    根据image_id连接Phase1和Phase2

    image_id是两个阶段的数据关联键
    """

    data = phase1.merge(
        phase2,
        on="image_id",
        how="left"
    )

    return data



def create_sample(row):
    """
    把csv的一行转换成统一condition格式
    """

    sample = {

        "image_id":
            row["image_id"],


        "image":
            row.get(
                "image_path",
                None
            ),


        "deca":
            row.get(
                "deca_path",
                None
            ),


        "phase2":
            row.get(
                "phase2_path",
                None
            ),


        "gaze_pitch":
            row.get(
                "gaze_pitch",
                None
            ),


        "gaze_yaw":
            row.get(
                "gaze_yaw",
                None
            )

    }


    return sample



def save_jsonl(
        data,
        output
):

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:


        for _,row in data.iterrows():

            sample=create_sample(row)


            f.write(
                json.dumps(sample)
                +
                "\n"
            )



def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--phase1",
        required=True
    )


    parser.add_argument(
        "--phase2",
        required=True
    )


    parser.add_argument(
        "--output",
        required=True
    )


    args=parser.parse_args()



    phase1=load_csv(
        args.phase1
    )


    phase2=load_csv(
        args.phase2
    )



    dataset=merge_data(
        phase1,
        phase2
    )


    print(
        "samples:",
        len(dataset)
    )


    save_jsonl(
        dataset,
        args.output
    )



if __name__=="__main__":
    main()