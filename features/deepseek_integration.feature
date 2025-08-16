Feature: DeepSeek AI Integration
  As a user
  I want to use DeepSeek AI for Chinese language processing
  So that I can get high-quality, cost-effective AI formatting and analysis

  Background:
    Given I have a test TSV file with Chinese words
    And I have a DeepSeek AI configuration file

  Scenario: Successfully format Chinese meanings using DeepSeek AI
    Given I have the sample TSV file "success_test.tsv" with content:
      """
      高维	gāowéi	(math.) higher dimensional
      安慰剂	ānwèijì	(pharm.) placebo; something said or done merely to soothe
      """
    And I have a DeepSeek AI config file "deepseek_success_config.yaml" with content:
      """
      global:
        default_provider: "deepseek"

      providers:
        deepseek:
          api_key: "valid-deepseek-key"
          base_url: "https://api.deepseek.com"

      features:
        meaning_formatter:
          multi_char:
            provider: "deepseek"
            model: "deepseek-chat"
            prompt: "ai_config/prompts/meaning_formatter_multi_char.md"
            temperature: 0.3
      """
    When I run the DeepSeek command with mocked success "anki-pleco-importer convert success_test.tsv --ai-config deepseek_success_config.yaml --use-ai-formatting --dry-run --verbose"
    Then the command should succeed
    And the output should contain "Formatted using DeepSeek AI"
    And the output should contain "Model: deepseek-chat"
    And the output should contain "Cost: $"
    And the output should show formatted meanings with domain markup

  Scenario: Generate AI fields using DeepSeek for etymology and structure
    Given I have the sample TSV file "etymology_test.tsv" with content:
      """
      化石燃料	huàshíránliào	fossil fuel
      """
    And I have a DeepSeek AI config file "deepseek_fields_config.yaml" with content:
      """
      global:
        default_provider: "deepseek"

      providers:
        deepseek:
          api_key: "valid-deepseek-key"
          base_url: "https://api.deepseek.com"

      features:
        field_generation:
          multi_char:
            provider: "deepseek"
            model: "deepseek-chat"
            prompt: "ai_config/prompts/field_generation_multi_char.md"
            temperature: 0.5
      """
    When I run the DeepSeek command with mocked field generation "anki-pleco-importer convert etymology_test.tsv --ai-config deepseek_fields_config.yaml --use-ai-fields --dry-run --verbose"
    Then the command should succeed
    And the output should contain "Generated fields using DeepSeek AI"
    And the output should contain "etymology_html"
    And the output should contain "structure_html"
    And the output should contain "Model: deepseek-chat"

  Scenario: Word classification using DeepSeek with custom temperature
    Given I have the sample TSV file "classify_test.tsv" with content:
      """
      测试词汇	cèshìcíhuì	test vocabulary
      """
    And I have a DeepSeek AI config file "deepseek_classifier_config.yaml" with content:
      """
      global:
        default_provider: "deepseek"

      providers:
        deepseek:
          api_key: "valid-deepseek-key"
          base_url: "https://api.deepseek.com"

      features:
        word_classifier:
          provider: "deepseek"
          model: "deepseek-chat"
          prompt: "ai_config/prompts/word_classifier.md"
          temperature: 0.1
      """
    When I run the DeepSeek command with mocked classification "anki-pleco-importer convert classify_test.tsv --ai-config deepseek_classifier_config.yaml --classify-words --dry-run --verbose"
    Then the command should succeed
    And the output should contain "Classified using DeepSeek AI"
    And the output should contain "Temperature: 0.1"
    And the output should contain "Classification: worth_learning"

  Scenario: DeepSeek API authentication failure with proper error handling
    Given I have the sample TSV file "fail_test.tsv" with content:
      """
      高维	gāowéi	(math.) higher dimensional
      """
    And I have a DeepSeek AI config file "deepseek_fail_config.yaml" with content:
      """
      global:
        default_provider: "deepseek"

      providers:
        deepseek:
          api_key: "invalid-key-format"
          base_url: "https://api.deepseek.com"

      features:
        meaning_formatter:
          multi_char:
            provider: "deepseek"
            model: "deepseek-chat"
            prompt: "ai_config/prompts/meaning_formatter_multi_char.md"
            temperature: 0.3
      """
    When I run the DeepSeek command with mocked failure "anki-pleco-importer convert fail_test.tsv --ai-config deepseek_fail_config.yaml --use-ai-formatting --dry-run --verbose"
    Then the command should fail with invalid API key
    And the output should contain "AI formatting failed"
    And the output should contain "Authentication Fails"

  Scenario: Cost tracking with DeepSeek pricing calculation
    Given I have the sample TSV file "cost_test.tsv" with content:
      """
      测试	cèshì	test
      价格	jiàgé	price
      """
    And I have a DeepSeek AI config file "deepseek_cost_config.yaml" with content:
      """
      global:
        default_provider: "deepseek"
        max_daily_cost_usd: 10.0

      providers:
        deepseek:
          api_key: "valid-deepseek-key"
          base_url: "https://api.deepseek.com"

      features:
        meaning_formatter:
          multi_char:
            provider: "deepseek"
            model: "deepseek-chat"
            prompt: "ai_config/prompts/meaning_formatter_multi_char.md"
            temperature: 0.3
      """
    When I run the DeepSeek command with mocked cost tracking "anki-pleco-importer convert cost_test.tsv --ai-config deepseek_cost_config.yaml --use-ai-formatting --dry-run --verbose"
    Then the command should succeed
    And the output should contain "Total AI cost: $"
    And the output should contain "DeepSeek pricing: $0.27 input"
    And the output should contain "2 words processed"
    And the cost should be calculated correctly for 2 words