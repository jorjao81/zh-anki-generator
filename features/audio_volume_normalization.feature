Feature: Audio Volume Normalization
  As a Chinese language learner
  I want all generated audio files to have consistent volume levels
  So that I have a better listening experience without volume adjustments

  Background:
    Given the audio generators are available
    And the target volume level is -15.0 dBFS

  Scenario: Forvo audio is normalized to target volume
    Given I have a Forvo API key
    And the word "你好" is available in Forvo
    When I generate audio using Forvo provider for "你好"
    Then the generated audio file should exist
    And the audio volume should be approximately -15.0 dBFS

  Scenario: Qwen TTS audio is normalized to target volume  
    Given I have a Qwen API key
    And the Qwen TTS service is available
    When I generate audio using Qwen provider for "你好"
    Then the generated audio file should exist
    And the audio volume should be approximately -15.0 dBFS

  Scenario: Tencent TTS audio is normalized to target volume
    Given I have Tencent API credentials
    And the Tencent TTS service is available  
    When I generate audio using Tencent provider for "你好"
    Then the generated audio file should exist
    And the audio volume should be approximately -15.0 dBFS

  Scenario: Multiple providers produce consistent volume levels
    Given I have API keys for multiple audio providers
    And the multi-provider audio generator is configured
    When I generate audio for the same word "测试" using different providers
    Then all generated audio files should have approximately -15.0 dBFS volume
    And the volume difference between files should be less than 2.0 dBFS

  Scenario: Cached audio files maintain normalized volume
    Given I have already generated normalized audio for "测试" 
    When I request the same audio again from cache
    Then the cached audio file should be returned
    And the audio volume should be approximately -15.0 dBFS

  Scenario: Volume normalization preserves audio quality
    Given I generate audio for "学习" using any provider
    When the audio is normalized to -15.0 dBFS
    Then the audio file should be playable
    And the audio should not contain clipping or distortion
    And the normalized audio should maintain the original pronunciation clarity