from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ..models.database import get_db, Block

router = APIRouter()

@router.get("/")
async def get_blocks(
    limit: int = Query(50, description="Number of blocks", le=1000),
    offset: int = Query(0, description="Offset"),
    db: Session = Depends(get_db)
):
    """Get blocks"""
    query = db.query(Block)
    total = query.count()
    
    blocks = query.order_by(Block.block_number.desc()).offset(offset).limit(limit).all()
    
    return {
        "blocks": [
            {
                "block_number": block.block_number,
                "block_hash": block.block_hash,
                "timestamp": block.timestamp,
                "gas_limit": block.gas_limit,
                "gas_used": block.gas_used,
                "miner": block.miner,
                "transaction_count": block.transaction_count,
                "size": block.size
            }
            for block in blocks
        ],
        "total": total
    }

@router.get("/{block_number}")
async def get_block(block_number: int, db: Session = Depends(get_db)):
    """Get specific block"""
    block = db.query(Block).filter(Block.block_number == block_number).first()
    
    if not block:
        raise HTTPException(status_code=404, detail="Block not found")
    
    return {
        "block_number": block.block_number,
        "block_hash": block.block_hash,
        "parent_hash": block.parent_hash,
        "timestamp": block.timestamp,
        "gas_limit": block.gas_limit,
        "gas_used": block.gas_used,
        "miner": block.miner,
        "difficulty": block.difficulty,
        "total_difficulty": block.total_difficulty,
        "size": block.size,
        "transaction_count": block.transaction_count,
        "base_fee_per_gas": block.base_fee_per_gas,
        "source": block.source,
        "created_at": block.created_at,
        "updated_at": block.updated_at
    }