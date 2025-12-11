
import asyncio
import os
import sys

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

from db.client import supabase_client

def check_status():
    document_id = "9d2347bd-0db8-488a-8fcc-abf34cb498ef"
    print(f"Checking status for document: {document_id}")
    
    try:
        response = supabase_client.table("documents").select("*").eq("id", document_id).execute()
        if response.data:
            doc = response.data[0]
            print(f"Document Found:")
            print(f"  ID: {doc.get('id')}")
            print(f"  Filename: {doc.get('filename')}")
            print(f"  Status: {doc.get('upload_status')}")
            print(f"  Error Message: {doc.get('error_message')}")
        else:
            print("Document not found.")
            
    except Exception as e:
        print(f"Error querying database: {e}")

if __name__ == "__main__":
    check_status()
