"""Reconcile an import's status without downloading or embedding it again."""
import argparse
import asyncio

import httpx

import main


async def retry(document_id):
    async with httpx.AsyncClient() as client:
        main.client = client
        return await main.send_callback({
            "document_id": document_id,
            "status": "completed",
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document_id", help="ID of an import whose vectors are fully stored")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(retry(args.document_id)) else 1)
