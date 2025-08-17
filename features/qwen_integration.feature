Feature: Qwen AI Integration
  As a user
  I want to use Qwen AI for Chinese language processing
  So that I can get high-quality, cost-effective AI formatting and analysis

  Background:
    Given I have a test TSV file with Chinese words
    And I have a Qwen AI configuration file

  Scenario: Successfully format Chinese meanings using Qwen AI
    Given I have the sample TSV file "success_test.tsv" with content:
      """
      高维	gāowéi	(math.) higher dimensional
      安慰剂	ānwèijì	(pharm.) placebo; something said or done merely to soothe
      """
    And I have a Qwen AI config file "qwen_success_config.yaml" with content:
      """
      global:
        default_provider: "qwen"

      providers:
        qwen:
          api_key: "valid-qwen-key"
          base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

      features:
        meaning_formatter:
          multi_char:
            provider: "qwen"
            model: "qwen-turbo"
            prompt: "ai_config/prompts/meaning_formatter_multi_char.md"
            temperature: 0.3
      """
    When I run the Qwen command with mocked success "anki-pleco-importer convert success_test.tsv --ai-config qwen_success_config.yaml --use-ai-formatting --dry-run --verbose"
    Then the command should succeed
    And the output should contain "Formatted using Qwen AI"
    And the output should contain "Model: qwen-turbo"
    And the output should contain "Cost: $"
    And the output should show formatted meanings with domain markup

  Scenario: Generate AI fields using Qwen for etymology and structure
    Given I have the sample TSV file "etymology_test.tsv" with content:
      """
      化石燃料	huàshíránliào	fossil fuel
      """
    And I have a Qwen AI config file "qwen_fields_config.yaml" with content:
      """
      global:
        default_provider: "qwen"

      providers:
        qwen:
          api_key: "valid-qwen-key"
          base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

      features:
        field_generation:
          multi_char:
            provider: "qwen"
            model: "qwen-turbo"
            prompt: "ai_config/prompts/field_generation_multi_char.md"
            temperature: 0.5
      """
    When I run the Qwen command with mocked field generation "anki-pleco-importer convert etymology_test.tsv --ai-config qwen_fields_config.yaml --use-ai-fields --dry-run --verbose"
    Then the command should succeed
    And the output should contain "Generated fields using Qwen AI"
    And the output should contain "etymology_html"
    And the output should contain "structure_html"
    And the output should contain "Model: qwen-turbo"

  Scenario: Word classification using Qwen with custom temperature
    Given I have the sample TSV file "classify_test.tsv" with content:
      """
      测试词汇	cèshìcíhuì	test vocabulary
      """
    And I have a Qwen AI config file "qwen_classifier_config.yaml" with content:
      """
      global:
        default_provider: "qwen"

      providers:
        qwen:
          api_key: "valid-qwen-key"
          base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

      features:
        word_classifier:
          provider: "qwen"
          model: "qwen-turbo"
          prompt: "ai_config/prompts/word_classifier.md"
          temperature: 0.1
      """
    When I run the Qwen command with mocked classification "anki-pleco-importer convert classify_test.tsv --ai-config qwen_classifier_config.yaml --classify-words --dry-run --verbose"
    Then the command should succeed
    And the output should contain "Classified using Qwen AI"
    And the output should contain "Temperature: 0.1"
    And the output should contain "Classification: worth_learning"

  Scenario: Qwen API authentication failure with proper error handling
    Given I have the sample TSV file "fail_test.tsv" with content:
      """
      高维	gāowéi	(math.) higher dimensional
      """
    And I have a Qwen AI config file "qwen_fail_config.yaml" with content:
      """
      global:
        default_provider: "qwen"

      providers:
        qwen:
          api_key: "invalid-key-format"
          base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

      features:
        meaning_formatter:
          multi_char:
            provider: "qwen"
            model: "qwen-turbo"
            prompt: "ai_config/prompts/meaning_formatter_multi_char.md"
            temperature: 0.3
      """
    When I run the Qwen command with mocked failure "anki-pleco-importer convert fail_test.tsv --ai-config qwen_fail_config.yaml --use-ai-formatting --dry-run --verbose"
    Then the command should fail with invalid API key
    And the output should contain "AI formatting failed"
    And the output should contain "Invalid API key"

  Scenario: Cost tracking with Qwen pricing calculation
    Given I have the sample TSV file "cost_test.tsv" with content:
      """
      测试	cèshì	test
      价格	jiàgé	price
      """
    And I have a Qwen AI config file "qwen_cost_config.yaml" with content:
      """
      global:
        default_provider: "qwen"
        max_daily_cost_usd: 10.0

      providers:
        qwen:
          api_key: "valid-qwen-key"
          base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

      features:
        meaning_formatter:
          multi_char:
            provider: "qwen"
            model: "qwen-turbo"
            prompt: "ai_config/prompts/meaning_formatter_multi_char.md"
            temperature: 0.3
      """
    When I run the Qwen command with mocked cost tracking "anki-pleco-importer convert cost_test.tsv --ai-config qwen_cost_config.yaml --use-ai-formatting --dry-run --verbose"
    Then the command should succeed
    And the output should contain "Total AI cost: $"
    And the output should contain "Qwen pricing: $0.086 input"
    And the output should contain "2 words processed"
    And the cost should be calculated correctly for 2 words

  Scenario: Test Qwen Max model with higher quality processing
    Given I have the sample TSV file "quality_test.tsv" with content:
      """
      复杂词汇	fùzácíhuì	complex vocabulary
      """
    And I have a Qwen AI config file "qwen_max_config.yaml" with content:
      """
      global:
        default_provider: "qwen"

      providers:
        qwen:
          api_key: "valid-qwen-key"
          base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

      features:
        meaning_formatter:
          multi_char:
            provider: "qwen"
            model: "qwen-max"
            prompt: "ai_config/prompts/meaning_formatter_multi_char.md"
            temperature: 0.2
      """
    When I run the Qwen command with mocked success "anki-pleco-importer convert quality_test.tsv --ai-config qwen_max_config.yaml --use-ai-formatting --dry-run --verbose"
    Then the command should succeed
    And the output should contain "Model: qwen-max"
    And the output should contain "Formatted using Qwen AI"
    And the cost should reflect qwen-max pricing