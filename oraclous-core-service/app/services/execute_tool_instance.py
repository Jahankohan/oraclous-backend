from app.core.database import get_session
from app.repositories.instance_repository import InstanceRepository
from app.services.credential_client import CredentialClient
from app.tools.factory import ToolFactory
from app.schemas.tool_instance import ExecutionContext

async def execute_tool_instance(instance_id: str, input_data: dict, job_id: str = "manual-job"):
    # 1. Fetch the ToolInstance
    async with get_session() as db:
        instance_repo = InstanceRepository(db)
        instance = await instance_repo.get_instance(instance_id)

        # 2. Fetch credentials using credential_mappings
        cred_client = CredentialClient()
        credentials = {}
        for cred_type, cred_identifier in instance.credential_mappings.items():
            if cred_type == "OAUTH_TOKEN":
                # You may need to know the provider and required scopes
                provider = cred_identifier  # or map from tool definition/config
                required_scopes = []
                credentials[cred_type] = await cred_client.get_runtime_token(
                    user_id=str(instance.user_id),
                    provider=provider,
                    required_scopes=required_scopes
                )
            else:
                # For other credential types, fetch the credential data
                credentials[cred_type] = await cred_client._get_credential_data(cred_identifier)

        # 3. Prepare ExecutionContext
        context = ExecutionContext(
            instance_id=instance.id,
            workflow_id=str(instance.workflow_id),
            user_id=str(instance.user_id),
            job_id=job_id,
            credentials=credentials,
            configuration=instance.configuration,
            settings=instance.settings
        )

        # 4. Execute the tool using the factory
        result = await ToolFactory.execute_tool(instance, input_data, context)
        print(result)
        return result
