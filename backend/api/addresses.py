from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from ..models.database import get_db, Address, Transaction

router = APIRouter()

@router.get("/")
async def get_addresses(
    limit: int = Query(50, description="Number of addresses to return", le=1000),
    offset: int = Query(0, description="Number of addresses to skip"),
    db: Session = Depends(get_db)
):
    """Get addresses"""
    query = db.query(Address)
    total = query.count()
    
    addresses = query.order_by(Address.updated_at.desc()).offset(offset).limit(limit).all()
    
    return {
        "addresses": [
            {
                "address": addr.address,
                "balance_eth": addr.balance_eth,
                "transaction_count": addr.transaction_count,
                "first_seen": addr.first_seen,
                "last_seen": addr.last_seen,
                "is_contract": addr.is_contract,
                "contract_name": addr.contract_name,
                "labels": addr.labels
            }
            for addr in addresses
        ],
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
    }

@router.get("/{address}")
async def get_address_details(address: str, db: Session = Depends(get_db)):
    """Get address details"""
    addr = db.query(Address).filter(Address.address == address.lower()).first()
    
    if not addr:
        raise HTTPException(status_code=404, detail="Address not found")
    
    return {
        "address": addr.address,
        "balance_wei": addr.balance_wei,
        "balance_eth": addr.balance_eth,
        "transaction_count": addr.transaction_count,
        "first_seen": addr.first_seen,
        "last_seen": addr.last_seen,
        "is_contract": addr.is_contract,
        "contract_name": addr.contract_name,
        "labels": addr.labels,
        "source": addr.source,
        "created_at": addr.created_at,
        "updated_at": addr.updated_at
    }

@router.get("/{address}/transactions")
async def get_address_transactions(
    address: str,
    limit: int = Query(50, description="Number of transactions", le=1000),
    offset: int = Query(0, description="Offset"),
    db: Session = Depends(get_db)
):
    """Get transactions for an address"""
    address = address.lower()
    
    query = db.query(Transaction).filter(
        (Transaction.from_address == address) | (Transaction.to_address == address)
    )
    
    total = query.count()
    transactions = query.order_by(Transaction.timestamp.desc()).offset(offset).limit(limit).all()
    
    return {
        "transactions": [
            {
                "hash": tx.hash,
                "block_number": tx.block_number,
                "timestamp": tx.timestamp,
                "from_address": tx.from_address,
                "to_address": tx.to_address,
                "value_eth": tx.value_eth,
                "tx_fee_eth": tx.tx_fee_eth,
                "is_error": tx.is_error,
                "direction": "incoming" if tx.to_address == address else "outgoing"
            }
            for tx in transactions
        ],
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
    }