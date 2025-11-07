import scientia_sdk as sct
from typing import List
from .atinary_client import client, SDLABS_GROUP_NAME

sdlabs_api_client = client()

def get_workstation(workstation_id):

    wst_api = sct.WorkstationApi(sdlabs_api_client)
    workstation = wst_api.workstation_get(workstation_id).object
 
    print("This is the workstation name: ", workstation.name)

    return wst_api



def show_workstations():
    
    # used to create workstations
    wst_api = sct.WorkstationApi(sdlabs_api_client)

    wsts: List[sct.WorkstationListObj] = wst_api.workstations_list(
        is_public=False, group_id=SDLABS_GROUP_NAME
    ).objects

    for wst in wsts:
        #if wst.name == "Pumpbot V1":
        print(wst)


