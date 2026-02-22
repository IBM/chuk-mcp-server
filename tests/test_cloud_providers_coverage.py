#!/usr/bin/env python3
"""
Comprehensive coverage tests for cloud provider modules.

Targets uncovered lines in:
- cloud/providers/__init__.py  (lines 36-37, 58-59)
- cloud/providers/aws.py       (lines 121, 130, 135-140, 156-161, 224, 239, 254, 256, 264-265)
- cloud/providers/azure.py     (lines 105, 109, 123-128, 159, 175, 189-190)
- cloud/providers/gcp.py       (lines 129, 135, 144-148, 162-169, 220, 234, 250, 265, 267, 275-276)
"""

import os
from unittest.mock import Mock, patch

from chuk_mcp_server.cloud.providers import register_all_providers
from chuk_mcp_server.cloud.providers.aws import AWSProvider, register_aws_provider
from chuk_mcp_server.cloud.providers.azure import AzureProvider, register_azure_provider
from chuk_mcp_server.cloud.providers.gcp import GCPProvider, register_gcp_provider

# ============================================================================
# providers/__init__.py  --  cover except-Exception branches (lines 36-37, 58-59)
# ============================================================================


class TestRegisterAllProvidersExceptionBranches:
    """Cover the generic Exception handlers for AWS and Edge registration."""

    def test_aws_registration_general_exception(self):
        """Lines 36-37: except Exception when registering AWS provider."""
        mock_registry = Mock()

        with (
            patch("chuk_mcp_server.cloud.providers.gcp.register_gcp_provider"),
            patch(
                "chuk_mcp_server.cloud.providers.aws.register_aws_provider",
                side_effect=RuntimeError("AWS runtime boom"),
            ),
            patch("chuk_mcp_server.cloud.providers.azure.register_azure_provider"),
            patch("chuk_mcp_server.cloud.providers.edge.register_edge_providers"),
            patch("chuk_mcp_server.cloud.providers.logger") as mock_logger,
        ):
            register_all_providers(mock_registry)
            mock_logger.error.assert_any_call("Error registering AWS provider: AWS runtime boom")

    def test_edge_registration_general_exception(self):
        """Lines 58-59: except Exception when registering Edge providers."""
        mock_registry = Mock()

        with (
            patch("chuk_mcp_server.cloud.providers.gcp.register_gcp_provider"),
            patch("chuk_mcp_server.cloud.providers.aws.register_aws_provider"),
            patch("chuk_mcp_server.cloud.providers.azure.register_azure_provider"),
            patch(
                "chuk_mcp_server.cloud.providers.edge.register_edge_providers",
                side_effect=RuntimeError("Edge runtime boom"),
            ),
            patch("chuk_mcp_server.cloud.providers.logger") as mock_logger,
        ):
            register_all_providers(mock_registry)
            mock_logger.error.assert_any_call("Error registering Edge providers: Edge runtime boom")


# ============================================================================
# aws.py  --  cover all missing branches and service configs
# ============================================================================


class TestAWSProviderCoverage:
    """Fill coverage gaps in AWSProvider."""

    # ---- get_environment_type: elastic_beanstalk / ec2 branch (line 121) ----

    @patch.dict(os.environ, {"AWS_BEANSTALK_APPLICATION_NAME": "my-app"}, clear=True)
    def test_environment_type_elastic_beanstalk(self):
        """Line 121 (elif _is_elastic_beanstalk / _is_ec2 -> production)."""
        provider = AWSProvider()
        assert provider.get_environment_type() == "production"

    @patch.dict(
        os.environ,
        {"AWS_REGION": "us-west-2"},
        clear=True,
    )
    def test_environment_type_ec2(self):
        """Line 121: EC2 branch returns production."""
        provider = AWSProvider()
        assert provider.get_environment_type() == "production"

    # ---- get_service_type: lambda ARM64 (line 130) ----

    @patch.dict(
        os.environ,
        {
            "AWS_LAMBDA_FUNCTION_NAME": "arm-func",
            "AWS_LAMBDA_RUNTIME_API": "arm64-runtime",
        },
        clear=True,
    )
    def test_service_type_lambda_arm64(self):
        """Line 130: SVC_LAMBDA_ARM64 branch."""
        provider = AWSProvider()
        assert provider.get_service_type() == "lambda_arm64"

    # ---- get_service_type: fargate (lines 135-140 partially) ----

    @patch.dict(
        os.environ,
        {"AWS_EXECUTION_ENV": "AWS_ECS_FARGATE"},
        clear=True,
    )
    def test_service_type_fargate_via_execution_env(self):
        """Lines 133-134: Fargate branch via AWS_EXECUTION_ENV."""
        provider = AWSProvider()
        assert provider.get_service_type() == "fargate"

    # ---- get_service_type: elastic beanstalk (lines 135-136) ----

    @patch.dict(
        os.environ,
        {"AWS_BEANSTALK_APPLICATION_NAME": "my-app"},
        clear=True,
    )
    def test_service_type_elastic_beanstalk(self):
        """Lines 135-136: Elastic Beanstalk branch."""
        provider = AWSProvider()
        assert provider.get_service_type() == "elastic_beanstalk"

    # ---- get_service_type: ec2 (lines 137-138) ----

    @patch.dict(
        os.environ,
        {"AWS_REGION": "us-east-1"},
        clear=True,
    )
    def test_service_type_ec2(self):
        """Lines 137-138: EC2 branch."""
        provider = AWSProvider()
        assert provider.get_service_type() == "ec2"

    # ---- get_service_type: generic (lines 139-140) ----

    @patch.dict(os.environ, {}, clear=True)
    def test_service_type_aws_generic(self):
        """Lines 139-140: aws_generic fallback."""
        provider = AWSProvider()
        assert provider.get_service_type() == "aws_generic"

    # ---- get_config_overrides: fargate config (line 155) ----

    @patch.dict(
        os.environ,
        {
            "ECS_CONTAINER_METADATA_URI": "http://169.254.170.2/v3",
            "ECS_CONTAINER_METADATA_URI_V4": "http://169.254.170.2/v4/abc",
        },
        clear=True,
    )
    def test_config_overrides_fargate_details(self):
        """Lines 154-155: Fargate config branch in get_config_overrides."""
        provider = AWSProvider()
        config = provider.get_config_overrides()
        assert config["cloud_provider"] == "aws"
        assert config["service_type"] == "fargate"
        assert config["performance_mode"] == "fargate_optimized"
        assert config["workers"] == 4
        assert config["max_connections"] == 2000
        assert config["task_arn"] == "http://169.254.170.2/v4/abc"

    # ---- get_config_overrides: beanstalk config (lines 156-161) ----

    @patch.dict(
        os.environ,
        {
            "AWS_BEANSTALK_APPLICATION_NAME": "my-app",
            "AWS_BEANSTALK_ENVIRONMENT_NAME": "my-env",
            "AWS_BEANSTALK_VERSION_LABEL": "v1.0",
        },
        clear=True,
    )
    def test_config_overrides_beanstalk(self):
        """Lines 156-157: Beanstalk config branch."""
        provider = AWSProvider()
        config = provider.get_config_overrides()
        assert config["cloud_provider"] == "aws"
        assert config["service_type"] == "elastic_beanstalk"
        assert config["performance_mode"] == "beanstalk_optimized"
        assert config["application_name"] == "my-app"
        assert config["environment_name"] == "my-env"
        assert config["version_label"] == "v1.0"
        assert config["workers"] == 4
        assert config["max_connections"] == 3000

    # ---- get_config_overrides: ec2 config (lines 158-159) ----

    @patch.dict(
        os.environ,
        {"AWS_REGION": "eu-west-1"},
        clear=True,
    )
    def test_config_overrides_ec2(self):
        """Lines 158-159: EC2 config branch."""
        provider = AWSProvider()
        config = provider.get_config_overrides()
        assert config["cloud_provider"] == "aws"
        assert config["service_type"] == "ec2"
        assert config["performance_mode"] == "ec2_optimized"
        assert config["workers"] == 4
        assert config["max_connections"] == 5000
        assert config["region"] == "eu-west-1"

    # ---- get_config_overrides: generic fallback (lines 160-161) ----

    @patch.dict(os.environ, {}, clear=True)
    def test_config_overrides_generic(self):
        """Lines 160-161: Generic fallback returning base_config only."""
        provider = AWSProvider()
        config = provider.get_config_overrides()
        assert config["cloud_provider"] == "aws"
        assert config["service_type"] == "aws_generic"
        # Generic has no extra keys like workers, host, etc.
        assert "workers" not in config
        assert "host" not in config

    # ---- _get_ec2_config (line 239, 224 etc.) ----

    @patch.dict(
        os.environ,
        {"AWS_REGION": "ap-southeast-1", "PORT": "9090"},
        clear=True,
    )
    def test_ec2_config_with_custom_port(self):
        """Cover _get_ec2_config with custom PORT env."""
        provider = AWSProvider()
        config = provider._get_ec2_config()
        assert config["host"] == "0.0.0.0"
        assert config["port"] == 9090
        assert config["debug"] is False
        assert config["log_level"] == "INFO"

    # ---- _get_beanstalk_config (line 224) ----

    @patch.dict(
        os.environ,
        {"AWS_BEANSTALK_APPLICATION_NAME": "bapp"},
        clear=True,
    )
    def test_beanstalk_config_defaults(self):
        """Cover _get_beanstalk_config default values."""
        provider = AWSProvider()
        config = provider._get_beanstalk_config()
        assert config["application_name"] == "bapp"
        assert config["environment_name"] == "unknown"
        assert config["version_label"] == "unknown"
        assert config["port"] == 8000

    # ---- Lambda performance modes (lines 254, 256) ----

    @patch.dict(
        os.environ,
        {"AWS_LAMBDA_FUNCTION_MEMORY_SIZE": "3008"},
        clear=True,
    )
    def test_lambda_performance_mode_high_memory(self):
        """Line 254: LAMBDA_MAX_MEMORY (3008) -> lambda_high_memory."""
        provider = AWSProvider()
        assert provider._get_lambda_performance_mode() == "lambda_high_memory"

    @patch.dict(
        os.environ,
        {"AWS_LAMBDA_FUNCTION_MEMORY_SIZE": "1024"},
        clear=True,
    )
    def test_lambda_performance_mode_standard(self):
        """Line 256: 1024 >= MEMORY_MEDIUM_THRESHOLD -> lambda_standard."""
        provider = AWSProvider()
        assert provider._get_lambda_performance_mode() == "lambda_standard"

    @patch.dict(
        os.environ,
        {"AWS_LAMBDA_FUNCTION_MEMORY_SIZE": "512"},
        clear=True,
    )
    def test_lambda_performance_mode_minimal(self):
        """Line 258: < 1024 -> lambda_minimal."""
        provider = AWSProvider()
        assert provider._get_lambda_performance_mode() == "lambda_minimal"

    # ---- register_aws_provider (lines 264-265) ----

    def test_register_aws_provider(self):
        """Lines 264-265: register_aws_provider creates and registers provider."""
        mock_registry = Mock()
        register_aws_provider(mock_registry)
        mock_registry.register_provider.assert_called_once()
        registered = mock_registry.register_provider.call_args[0][0]
        assert isinstance(registered, AWSProvider)

    # ---- _is_ec2 (line 177-184) ----

    @patch.dict(os.environ, {"AWS_REGION": "us-east-1"}, clear=True)
    def test_is_ec2_true(self):
        """EC2 detected when AWS_REGION set but not lambda/fargate/beanstalk."""
        provider = AWSProvider()
        assert provider._is_ec2() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_ec2_false_no_region(self):
        """EC2 not detected without AWS_REGION."""
        provider = AWSProvider()
        assert provider._is_ec2() is False

    # ---- _get_region fallback ----

    @patch.dict(os.environ, {}, clear=True)
    def test_get_region_default(self):
        """Region defaults to us-east-1."""
        provider = AWSProvider()
        assert provider._get_region() == "us-east-1"

    @patch.dict(os.environ, {"AWS_DEFAULT_REGION": "eu-central-1"}, clear=True)
    def test_get_region_from_default_region(self):
        """Region from AWS_DEFAULT_REGION when AWS_REGION is absent."""
        provider = AWSProvider()
        assert provider._get_region() == "eu-central-1"

    # ---- Lambda config with high memory ----

    @patch.dict(
        os.environ,
        {
            "AWS_LAMBDA_FUNCTION_NAME": "high-mem-fn",
            "AWS_LAMBDA_FUNCTION_MEMORY_SIZE": "3008",
            "AWS_LAMBDA_FUNCTION_TIMEOUT": "300",
            "AWS_LAMBDA_FUNCTION_VERSION": "42",
            "AWS_LAMBDA_RUNTIME_API": "arm64-runtime",
        },
        clear=True,
    )
    def test_lambda_config_high_memory_details(self):
        """Cover _get_lambda_config with high memory."""
        provider = AWSProvider()
        config = provider.get_config_overrides()
        assert config["service_type"] == "lambda_arm64"
        assert config["memory_mb"] == 3008
        assert config["timeout_sec"] == 300
        assert config["function_version"] == "42"
        assert config["performance_mode"] == "lambda_high_memory"
        assert config["runtime"] == "arm64-runtime"


# ============================================================================
# azure.py  --  cover all missing branches
# ============================================================================


class TestAzureProviderCoverage:
    """Fill coverage gaps in AzureProvider."""

    # ---- get_service_type: App Service branch (line 105) ----

    @patch.dict(
        os.environ,
        {"WEBSITE_SITE_NAME": "my-site"},
        clear=True,
    )
    def test_service_type_app_service(self):
        """Line 104-105: App Service branch (not azure_functions, not ACI)."""
        provider = AzureProvider()
        assert provider.get_service_type() == "app_service"

    # ---- get_service_type: azure_functions with runtime (line 103) ----

    @patch.dict(
        os.environ,
        {
            "AZURE_FUNCTIONS_ENVIRONMENT": "Production",
            "FUNCTIONS_WORKER_RUNTIME": "node",
        },
        clear=True,
    )
    def test_service_type_functions_with_runtime(self):
        """Line 103: azure_functions_{runtime} branch."""
        provider = AzureProvider()
        assert provider.get_service_type() == "azure_functions_node"

    # ---- get_service_type: azure_functions without runtime (line 103 else) ----

    @patch.dict(
        os.environ,
        {"AZURE_FUNCTIONS_ENVIRONMENT": "Development"},
        clear=True,
    )
    def test_service_type_functions_no_runtime(self):
        """Line 103: plain azure_functions when no runtime set."""
        provider = AzureProvider()
        assert provider.get_service_type() == "azure_functions"

    # ---- get_service_type: generic azure (line 109) ----

    @patch.dict(os.environ, {}, clear=True)
    def test_service_type_azure_generic(self):
        """Line 108-109: azure_generic fallback."""
        provider = AzureProvider()
        assert provider.get_service_type() == "azure_generic"

    # ---- get_config_overrides: app_service config (lines 123-128) ----

    @patch.dict(
        os.environ,
        {
            "WEBSITE_SITE_NAME": "my-webapp",
            "WEBSITE_RESOURCE_GROUP": "rg-prod",
            "WEBSITE_SKU": "P1v3",
            "WEBSITE_INSTANCE_ID": "inst-123",
        },
        clear=True,
    )
    def test_config_overrides_app_service(self):
        """Lines 123-124: App Service config branch."""
        provider = AzureProvider()
        config = provider.get_config_overrides()
        assert config["cloud_provider"] == "azure"
        assert config["service_type"] == "app_service"
        assert config["performance_mode"] == "app_service_optimized"
        assert config["site_name"] == "my-webapp"
        assert config["resource_group"] == "rg-prod"
        assert config["sku"] == "P1v3"
        assert config["instance_id"] == "inst-123"
        assert config["workers"] == 4
        assert config["max_connections"] == 2000

    # ---- get_config_overrides: container_instances config (lines 125-126) ----

    @patch.dict(
        os.environ,
        {"ACI_RESOURCE_GROUP": "aci-rg"},
        clear=True,
    )
    def test_config_overrides_container_instances(self):
        """Lines 125-126: Container Instances config branch."""
        provider = AzureProvider()
        config = provider.get_config_overrides()
        assert config["cloud_provider"] == "azure"
        assert config["service_type"] == "container_instances"
        assert config["performance_mode"] == "aci_optimized"
        assert config["resource_group"] == "aci-rg"
        assert config["workers"] == 2
        assert config["max_connections"] == 1000

    # ---- get_config_overrides: generic fallback (lines 127-128) ----

    @patch.dict(os.environ, {}, clear=True)
    def test_config_overrides_azure_generic(self):
        """Lines 127-128: Generic fallback returning base_config only."""
        provider = AzureProvider()
        config = provider.get_config_overrides()
        assert config["cloud_provider"] == "azure"
        assert config["service_type"] == "azure_generic"
        assert "workers" not in config
        assert "host" not in config

    # ---- _get_app_service_config details (line 159) ----

    @patch.dict(
        os.environ,
        {"WEBSITE_SITE_NAME": "s", "PORT": "5000"},
        clear=True,
    )
    def test_app_service_config_custom_port(self):
        """Line 159: App Service config with custom PORT."""
        provider = AzureProvider()
        config = provider._get_app_service_config()
        assert config["port"] == 5000
        assert config["host"] == "0.0.0.0"
        assert config["debug"] is False
        assert config["log_level"] == "INFO"

    # ---- _get_container_instances_config details (line 175) ----

    @patch.dict(
        os.environ,
        {"ACI_RESOURCE_GROUP": "rg-aci", "PORT": "3000"},
        clear=True,
    )
    def test_container_instances_config_custom_port(self):
        """Line 175: Container Instances config with custom PORT."""
        provider = AzureProvider()
        config = provider._get_container_instances_config()
        assert config["port"] == 3000
        assert config["host"] == "0.0.0.0"
        assert config["resource_group"] == "rg-aci"

    # ---- register_azure_provider (lines 189-190) ----

    def test_register_azure_provider(self):
        """Lines 189-190: register_azure_provider creates and registers provider."""
        mock_registry = Mock()
        register_azure_provider(mock_registry)
        mock_registry.register_provider.assert_called_once()
        registered = mock_registry.register_provider.call_args[0][0]
        assert isinstance(registered, AzureProvider)

    # ---- environment type: container instances serverless ----

    @patch.dict(
        os.environ,
        {"ACI_RESOURCE_GROUP": "rg-aci"},
        clear=True,
    )
    def test_environment_type_aci_serverless(self):
        """Container Instances returns serverless env type."""
        provider = AzureProvider()
        assert provider.get_environment_type() == "serverless"

    # ---- azure functions config details ----

    @patch.dict(
        os.environ,
        {
            "AZURE_FUNCTIONS_ENVIRONMENT": "Production",
            "FUNCTIONS_WORKER_RUNTIME": "python",
            "FUNCTIONS_EXTENSION_VERSION": "~4",
            "AzureWebJobsScriptRoot": "/home/site/wwwroot",
        },
        clear=True,
    )
    def test_azure_functions_config_details(self):
        """Cover _get_azure_functions_config fields."""
        provider = AzureProvider()
        config = provider._get_azure_functions_config()
        assert config["port"] == 7071
        assert config["workers"] == 1
        assert config["max_connections"] == 200
        assert config["worker_runtime"] == "python"
        assert config["extension_version"] == "~4"
        assert config["script_root"] == "/home/site/wwwroot"

    # ---- detect via weak indicators ----

    @patch.dict(
        os.environ,
        {"AZURE_CLIENT_ID": "abc", "AZURE_SUBSCRIPTION_ID": "sub-123"},
        clear=True,
    )
    def test_detect_weak_indicators_client_and_subscription(self):
        """Detect Azure via 2 weak indicators: CLIENT_ID + SUBSCRIPTION_ID."""
        provider = AzureProvider()
        assert provider.detect() is True

    @patch.dict(
        os.environ,
        {"AZURE_CLIENT_ID": "abc"},
        clear=True,
    )
    def test_detect_single_weak_indicator_not_enough(self):
        """A single weak indicator is not sufficient for detection."""
        provider = AzureProvider()
        assert provider.detect() is False


# ============================================================================
# gcp.py  --  cover all missing branches
# ============================================================================


class TestGCPProviderCoverage:
    """Fill coverage gaps in GCPProvider."""

    # ---- get_environment_type: generic production (line 129) ----

    @patch.dict(os.environ, {"GCP_PROJECT": "my-proj"}, clear=True)
    def test_environment_type_generic_production(self):
        """Line 128-129: Generic GCP -> production fallback."""
        provider = GCPProvider()
        assert provider.get_environment_type() == "production"

    # ---- get_environment_type: compute engine production (line 127) ----

    @patch.dict(os.environ, {"GCE_METADATA_TIMEOUT": "5"}, clear=True)
    def test_environment_type_compute_engine(self):
        """Line 126-127: Compute Engine -> production."""
        provider = GCPProvider()
        assert provider.get_environment_type() == "production"

    # ---- get_service_type: GCF Gen1 (line 135) ----

    @patch.dict(
        os.environ,
        {"GOOGLE_CLOUD_FUNCTION_NAME": "gen1-func"},
        clear=True,
    )
    def test_service_type_gcf_gen1(self):
        """Line 134-135: GCF Gen1 branch (GOOGLE_CLOUD_FUNCTION_NAME set)."""
        provider = GCPProvider()
        assert provider.get_service_type() == "gcf_gen1"

    # ---- get_service_type: GCF Gen2 (already tested, but exercise path) ----

    @patch.dict(
        os.environ,
        {"FUNCTION_TARGET": "my_handler"},
        clear=True,
    )
    def test_service_type_gcf_gen2_via_target(self):
        """GCF Gen2 via FUNCTION_TARGET."""
        provider = GCPProvider()
        assert provider.get_service_type() == "gcf_gen2"

    # ---- get_service_type: Cloud Run (lines 138-139) already tested ----

    # ---- get_service_type: App Engine flexible (lines 143-144) ----

    @patch.dict(
        os.environ,
        {"GAE_APPLICATION": "my-app", "GAE_ENV": "flexible"},
        clear=True,
    )
    def test_service_type_gae_flexible(self):
        """Lines 143-144: App Engine flexible branch."""
        provider = GCPProvider()
        assert provider.get_service_type() == "gae_flexible"

    @patch.dict(
        os.environ,
        {"GAE_APPLICATION": "my-app"},
        clear=True,
    )
    def test_service_type_gae_flexible_default(self):
        """GAE_ENV not set defaults to flexible."""
        provider = GCPProvider()
        assert provider.get_service_type() == "gae_flexible"

    # ---- get_service_type: Compute Engine (lines 145-146) ----

    @patch.dict(
        os.environ,
        {"GCE_METADATA_TIMEOUT": "5"},
        clear=True,
    )
    def test_service_type_compute_engine(self):
        """Lines 145-146: Compute Engine (GCE) branch."""
        provider = GCPProvider()
        assert provider.get_service_type() == "gce"

    # ---- get_service_type: generic (lines 147-148) ----

    @patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "proj"}, clear=True)
    def test_service_type_gcp_generic(self):
        """Lines 147-148: gcp_generic fallback."""
        provider = GCPProvider()
        assert provider.get_service_type() == "gcp_generic"

    # ---- get_config_overrides: Cloud Run config (lines 162-163) ----

    @patch.dict(
        os.environ,
        {
            "K_SERVICE": "my-svc",
            "K_REVISION": "rev-001",
            "CLOUD_RUN_CPU": "2",
            "CLOUD_RUN_CONCURRENCY": "500",
        },
        clear=True,
    )
    def test_config_overrides_cloud_run(self):
        """Lines 162-163: Cloud Run config branch."""
        provider = GCPProvider()
        config = provider.get_config_overrides()
        assert config["cloud_provider"] == "gcp"
        assert config["service_type"] == "cloud_run"
        assert config["performance_mode"] == "cloud_run_optimized"
        assert config["service_name"] == "my-svc"
        assert config["revision"] == "rev-001"
        assert config["workers"] == 4  # min(2*2, 8) = 4
        assert config["max_connections"] == 500

    # ---- get_config_overrides: App Engine config (lines 164-165) ----

    @patch.dict(
        os.environ,
        {
            "GAE_APPLICATION": "gae-app",
            "GAE_ENV": "standard",
            "GAE_SERVICE": "default",
            "GAE_VERSION": "v2",
            "GAE_RUNTIME": "python39",
        },
        clear=True,
    )
    def test_config_overrides_app_engine_standard(self):
        """Lines 164-165: App Engine config branch (standard)."""
        provider = GCPProvider()
        config = provider.get_config_overrides()
        assert config["cloud_provider"] == "gcp"
        assert config["service_type"] == "gae_standard"
        assert config["performance_mode"] == "app_engine_optimized"
        assert config["gae_application"] == "gae-app"
        assert config["gae_service"] == "default"
        assert config["gae_version"] == "v2"
        assert config["gae_runtime"] == "python39"
        assert config["workers"] == 1  # standard -> 1 worker

    @patch.dict(
        os.environ,
        {
            "GAE_APPLICATION": "gae-app",
            "GAE_ENV": "flexible",
        },
        clear=True,
    )
    def test_config_overrides_app_engine_flexible(self):
        """App Engine flexible config: workers == 4."""
        provider = GCPProvider()
        config = provider.get_config_overrides()
        assert config["service_type"] == "gae_flexible"
        assert config["workers"] == 4

    # ---- get_config_overrides: Compute Engine config (lines 166-167) ----

    @patch.dict(
        os.environ,
        {"GCE_METADATA_TIMEOUT": "10"},
        clear=True,
    )
    def test_config_overrides_compute_engine(self):
        """Lines 166-167: Compute Engine config branch."""
        provider = GCPProvider()
        config = provider.get_config_overrides()
        assert config["cloud_provider"] == "gcp"
        assert config["service_type"] == "gce"
        assert config["performance_mode"] == "gce_optimized"
        assert config["workers"] == 4
        assert config["max_connections"] == 5000

    # ---- get_config_overrides: generic fallback (lines 168-169) ----

    @patch.dict(
        os.environ,
        {"GOOGLE_CLOUD_PROJECT": "proj"},
        clear=True,
    )
    def test_config_overrides_gcp_generic(self):
        """Lines 168-169: Generic fallback returning base_config only."""
        provider = GCPProvider()
        config = provider.get_config_overrides()
        assert config["cloud_provider"] == "gcp"
        assert config["service_type"] == "gcp_generic"
        assert "workers" not in config
        assert "host" not in config

    # ---- _get_cloud_run_config details (line 220) ----

    @patch.dict(
        os.environ,
        {"K_SERVICE": "svc", "PORT": "9090"},
        clear=True,
    )
    def test_cloud_run_config_custom_port(self):
        """Line 220: Cloud Run config with custom PORT."""
        provider = GCPProvider()
        config = provider._get_cloud_run_config()
        assert config["port"] == 9090
        assert config["host"] == "0.0.0.0"
        assert config["log_level"] == "INFO"
        assert config["debug"] is False

    # ---- _get_app_engine_config details (line 234) ----

    @patch.dict(
        os.environ,
        {"GAE_APPLICATION": "app", "PORT": "4000"},
        clear=True,
    )
    def test_app_engine_config_custom_port(self):
        """Line 234: App Engine config with custom PORT."""
        provider = GCPProvider()
        config = provider._get_app_engine_config()
        assert config["port"] == 4000

    # ---- _get_compute_engine_config details (line 250) ----

    @patch.dict(
        os.environ,
        {"GCE_METADATA_TIMEOUT": "1", "PORT": "7070"},
        clear=True,
    )
    def test_compute_engine_config_custom_port(self):
        """Line 250: Compute Engine config with custom PORT."""
        provider = GCPProvider()
        config = provider._get_compute_engine_config()
        assert config["port"] == 7070
        assert config["host"] == "0.0.0.0"
        assert config["workers"] == 4
        assert config["max_connections"] == 5000
        assert config["debug"] is False
        assert config["log_level"] == "INFO"

    # ---- _get_gcf_performance_mode (lines 265, 267) ----

    @patch.dict(
        os.environ,
        {"FUNCTION_MEMORY_MB": "4096"},
        clear=True,
    )
    def test_gcf_performance_mode_high_memory(self):
        """Line 264-265: >= 2048 -> gcf_high_memory."""
        provider = GCPProvider()
        assert provider._get_gcf_performance_mode() == "gcf_high_memory"

    @patch.dict(
        os.environ,
        {"FUNCTION_MEMORY_MB": "2048"},
        clear=True,
    )
    def test_gcf_performance_mode_high_memory_exact(self):
        """Exactly 2048 -> gcf_high_memory."""
        provider = GCPProvider()
        assert provider._get_gcf_performance_mode() == "gcf_high_memory"

    @patch.dict(
        os.environ,
        {"FUNCTION_MEMORY_MB": "1024"},
        clear=True,
    )
    def test_gcf_performance_mode_standard(self):
        """Line 266-267: >= 1024 -> gcf_standard."""
        provider = GCPProvider()
        assert provider._get_gcf_performance_mode() == "gcf_standard"

    @patch.dict(
        os.environ,
        {"FUNCTION_MEMORY_MB": "256"},
        clear=True,
    )
    def test_gcf_performance_mode_minimal(self):
        """Line 268-269: < 1024 -> gcf_minimal."""
        provider = GCPProvider()
        assert provider._get_gcf_performance_mode() == "gcf_minimal"

    @patch.dict(os.environ, {}, clear=True)
    def test_gcf_performance_mode_default(self):
        """Default memory (512) -> gcf_minimal."""
        provider = GCPProvider()
        assert provider._get_gcf_performance_mode() == "gcf_minimal"

    # ---- register_gcp_provider (lines 275-276) ----

    def test_register_gcp_provider(self):
        """Lines 275-276: register_gcp_provider creates and registers provider."""
        mock_registry = Mock()
        register_gcp_provider(mock_registry)
        mock_registry.register_provider.assert_called_once()
        registered = mock_registry.register_provider.call_args[0][0]
        assert isinstance(registered, GCPProvider)

    # ---- _get_project_id fallbacks ----

    @patch.dict(os.environ, {"GCLOUD_PROJECT": "proj-2"}, clear=True)
    def test_get_project_id_gcloud(self):
        """Project ID via GCLOUD_PROJECT."""
        provider = GCPProvider()
        assert provider._get_project_id() == "proj-2"

    @patch.dict(os.environ, {"GCP_PROJECT": "proj-3"}, clear=True)
    def test_get_project_id_gcp_project(self):
        """Project ID via GCP_PROJECT."""
        provider = GCPProvider()
        assert provider._get_project_id() == "proj-3"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_project_id_default(self):
        """Project ID defaults to 'unknown'."""
        provider = GCPProvider()
        assert provider._get_project_id() == "unknown"

    # ---- Cloud Functions config (line 215 - gen1 function name fallback) ----

    @patch.dict(
        os.environ,
        {
            "GOOGLE_CLOUD_FUNCTION_NAME": "gen1-func",
            "FUNCTION_MEMORY_MB": "2048",
            "FUNCTION_TIMEOUT_SEC": "120",
        },
        clear=True,
    )
    def test_cloud_functions_config_gen1(self):
        """Cover _get_cloud_functions_config for Gen1 (line 215 fallback)."""
        provider = GCPProvider()
        config = provider._get_cloud_functions_config()
        assert config["function_name"] == "gen1-func"
        assert config["memory_mb"] == 2048
        assert config["timeout_sec"] == 120
        assert config["performance_mode"] == "gcf_high_memory"

    # ---- Cloud Run config with CPU scaling ----

    @patch.dict(
        os.environ,
        {"K_SERVICE": "svc", "CLOUD_RUN_CPU": "4"},
        clear=True,
    )
    def test_cloud_run_config_high_cpu(self):
        """Cloud Run workers capped at 8: min(4*2, 8) = 8."""
        provider = GCPProvider()
        config = provider._get_cloud_run_config()
        assert config["workers"] == 8

    @patch.dict(
        os.environ,
        {"K_SERVICE": "svc", "CLOUD_RUN_CPU": "8"},
        clear=True,
    )
    def test_cloud_run_config_very_high_cpu(self):
        """Cloud Run workers capped at 8: min(8*2, 8) = 8."""
        provider = GCPProvider()
        config = provider._get_cloud_run_config()
        assert config["workers"] == 8

    # ---- _is_app_engine ----

    @patch.dict(os.environ, {"GAE_APPLICATION": "test"}, clear=True)
    def test_is_app_engine_true(self):
        provider = GCPProvider()
        assert provider._is_app_engine() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_app_engine_false(self):
        provider = GCPProvider()
        assert provider._is_app_engine() is False

    # ---- _is_compute_engine ----

    @patch.dict(os.environ, {"GCE_METADATA_TIMEOUT": "1"}, clear=True)
    def test_is_compute_engine_true(self):
        provider = GCPProvider()
        assert provider._is_compute_engine() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_compute_engine_false(self):
        provider = GCPProvider()
        assert provider._is_compute_engine() is False

    # ---- environment type: cloud run + app engine -> serverless ----

    @patch.dict(os.environ, {"K_SERVICE": "svc"}, clear=True)
    def test_environment_type_cloud_run_serverless(self):
        provider = GCPProvider()
        assert provider.get_environment_type() == "serverless"

    @patch.dict(os.environ, {"GAE_APPLICATION": "app"}, clear=True)
    def test_environment_type_app_engine_serverless(self):
        provider = GCPProvider()
        assert provider.get_environment_type() == "serverless"
