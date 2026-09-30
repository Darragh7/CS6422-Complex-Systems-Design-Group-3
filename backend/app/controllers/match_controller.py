from fastapi import APIRouter

from app.services.football_api import FootballAPI


router = APIRouter(
    prefix="/matches",
    tags=["Matches"]
)


@router.get("/")
async def get_matches():
    football_api = FootballAPI()

    matches = await football_api.get_matches()

    return {
        "matches": matches
    }