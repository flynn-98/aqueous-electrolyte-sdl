
from .atinary_client import client, SDLABS_GROUP_NAME
import scientia_sdk as sct

sdlabs_api_client = client()

def launch_campaign(template_id):
    tpl_api = sct.TemplateApi(sdlabs_api_client)
    # Campaign run
    campaign_ids = []
    campaign_id = tpl_api.template_run(
        template_id, template_run_obj=sct.TemplateRunObj(preload_data=False)
    ).object.id


    print(
        f"Running new optimization with campaign id: {campaign_id}, associated to template {template_id}."
    )

    campaign_ids.append(campaign_id)
    return campaign_id


def get_campaign_ids_by_state(template_id: str, state: str = None):
    cpg_api = sct.CampaignApi(sdlabs_api_client)

    # Get states for that template (each state object has campaigns list)
    states = cpg_api.campaigns_state(
        template_ids=[template_id], group_id=SDLABS_GROUP_NAME
    ).objects

    if not states:
        return [], []

    # Filter states by the requested state (or all if none)
    r_states = [st for st in states if (state is None or st.state == state)]

    # From these, collect campaign IDs & names
    running_templates_id = []
    running_templates_name = []
    for st in r_states:
        for c in st.campaigns:
            running_templates_id.append(c.id)
            running_templates_name.append(c.name)

    # Optionally, you might want to verify or print only those campaigns that still exist
    all_campaigns = cpg_api.campaigns_list(group_id=SDLABS_GROUP_NAME).objects
    for c in all_campaigns:
        if c.id in running_templates_id:
            print(f"In use: {c.name} (id={c.id})")

    return running_templates_id, running_templates_name

