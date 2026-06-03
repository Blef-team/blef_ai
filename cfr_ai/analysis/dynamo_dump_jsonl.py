#!/usr/bin/env python3
"""READ-ONLY dump of the `games` DynamoDB table to a single gzipped JSONL file.

SAFETY: this calls ONLY DynamoDB Scan + DescribeTable. It performs NO writes of
any kind (no put_item / update_item / delete_item / batch_write). Nothing in the
table can be modified or deleted by this script.

Paced to a target RCU (default 5 = the table's 50% autoscale target, so it stays
within the provisioned 10 RCU and leaves headroom for live games). Eventually-
consistent reads (half the RCU cost). Resumable via a .ckpt file.
"""
import os
import sys
import time
import json
import gzip
import base64
import argparse
from decimal import Decimal

import boto3
from boto3.dynamodb.types import TypeDeserializer
from botocore.config import Config


def json_default(o):
    if isinstance(o, Decimal):
        return int(o) if o % 1 == 0 else float(o)
    if isinstance(o, (set, frozenset)):
        return list(o)
    if isinstance(o, (bytes, bytearray)):
        return base64.b64encode(o).decode()
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="games")
    ap.add_argument("--region", default="eu-west-2")
    ap.add_argument("--profile", default="a")
    ap.add_argument("--out", default="cfr_ai/analysis/data/games_current.jsonl.gz")
    ap.add_argument("--rcu", type=float, default=5.0)
    ap.add_argument("--page-size", type=int, default=60)
    args = ap.parse_args()

    ckpt = args.out + ".ckpt"
    cfg = Config(retries={"max_attempts": 12, "mode": "adaptive"})
    sess = boto3.Session(profile_name=args.profile, region_name=args.region)
    client = sess.client("dynamodb", config=cfg)
    deser = TypeDeserializer()

    approx = client.describe_table(TableName=args.table)["Table"].get("ItemCount", 0)
    print(f"target table={args.table} approx_items={approx} rcu={args.rcu} -> {args.out}", flush=True)

    scan_kwargs = {"TableName": args.table, "Limit": args.page_size,
                   "ConsistentRead": False, "ReturnConsumedCapacity": "TOTAL"}
    mode = "wt"
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            lek = json.load(f)
        if lek:
            scan_kwargs["ExclusiveStartKey"] = lek
            mode = "at"
            print("resuming from checkpoint", flush=True)

    total = 0
    pages = 0
    start = time.time()
    with gzip.open(args.out, mode, encoding="utf-8") as out:
        while True:
            resp = client.scan(**scan_kwargs)            # <-- READ ONLY
            pages += 1
            for item in resp.get("Items", []):
                clean = {k: deser.deserialize(v) for k, v in item.items()}
                out.write(json.dumps(clean, ensure_ascii=False, default=json_default) + "\n")
                total += 1
            out.flush()
            consumed = resp.get("ConsumedCapacity", {}).get("CapacityUnits", 0.5)
            last = resp.get("LastEvaluatedKey")
            with open(ckpt, "w") as f:
                json.dump(last, f, default=json_default)
            if pages % 20 == 0:
                el = time.time() - start
                pct = (100 * total / approx) if approx else 0
                print(f"  {total} items ({pct:4.1f}%) | {pages} pages | {el:5.0f}s | "
                      f"{total/el:4.1f}/s | page consumed {consumed:.1f} RCU", flush=True)
            if not last:
                break
            scan_kwargs["ExclusiveStartKey"] = last
            time.sleep(consumed / args.rcu)

    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"DONE: {total} items in {time.time()-start:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
