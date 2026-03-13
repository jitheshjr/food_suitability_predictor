from rest_framework.decorators import api_view
from rest_framework.response import Response
from agent.agent_service import run_agent


@api_view(['GET'])
def test_api(request):
    return Response({"message": "API working"})


@api_view(['POST'])
def predict(request):

    try:
        result = run_agent(request.data)
        return Response(result)

    except Exception as e:
        return Response({
            "error": str(e)
        }, status=500)