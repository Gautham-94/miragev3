"""Read endpoints for QueryMatch rows -- the "match found" results the Queries page
lists, produced by mirage.openvocab.dispatcher.OpenVocabDispatcher when a saved
open-vocabulary query matches a confirmed tracked object (TODO_FIX_LIST.md items 4/6).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from mirage.api.schemas import QueryMatchOut
from mirage.db.models import QueryMatch
from mirage.util.time import utc_from_timestamp

router = APIRouter(prefix="/api/query-matches", tags=["query-matches"])


@router.get("", response_model=list[QueryMatchOut])
def list_query_matches(
    camera: str | None = None,
    query_id: str | None = None,
    after: float | None = Query(None, description="epoch seconds, inclusive lower bound on matched_at"),
    before: float | None = Query(None, description="epoch seconds, exclusive upper bound on matched_at"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[QueryMatchOut]:
    q = QueryMatch.select()
    if camera is not None:
        q = q.where(QueryMatch.camera == camera)
    if query_id is not None:
        q = q.where(QueryMatch.query_id == query_id)
    if after is not None:
        q = q.where(QueryMatch.matched_at >= utc_from_timestamp(after))
    if before is not None:
        q = q.where(QueryMatch.matched_at < utc_from_timestamp(before))

    q = q.order_by(QueryMatch.matched_at.desc()).limit(limit).offset(offset)
    return [QueryMatchOut.from_model(m) for m in q]


@router.get("/{match_id}", response_model=QueryMatchOut)
def get_query_match(match_id: str) -> QueryMatchOut:
    match = QueryMatch.get_or_none(QueryMatch.id == match_id)
    if match is None:
        raise HTTPException(status_code=404, detail=f"unknown query match {match_id!r}")
    return QueryMatchOut.from_model(match)


@router.get("/{match_id}/thumbnail")
def get_query_match_thumbnail(match_id: str) -> FileResponse:
    match = QueryMatch.get_or_none(QueryMatch.id == match_id)
    if match is None:
        raise HTTPException(status_code=404, detail=f"unknown query match {match_id!r}")
    if not match.thumb_path:
        raise HTTPException(status_code=404, detail="this match has no thumbnail")
    path = Path(match.thumb_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail=f"thumbnail file no longer exists on disk: {match.thumb_path}")
    return FileResponse(path, media_type="image/jpeg")
