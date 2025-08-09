Feature: Chinese 2 note type parsing
  As a Chinese language learner
  I want Chinese 2 note type cards to be parsed correctly 
  So that I can see all my vocabulary including words with the new format

  Background:
    Given the AnkiExportParser is available

  Scenario: Parse Chinese 2 note type with simplified hanzi first
    Given I have an Anki export file with Chinese 2 note type data:
      """
      Chinese 2	空	kong1	[sound:kong1.mp3]	empty; sky
      """
    When I parse the file
    Then I should get 1 card
    And the card should have:
      | field       | value      |
      | notetype    | Chinese 2  |
      | characters  | 空         |
      | pinyin      | kong1      |
      | audio       | [sound:kong1.mp3] |
      | definitions | empty; sky |
    And the clean characters should be "空"

  Scenario: Parse mixed Chinese and Chinese 2 note types
    Given I have an Anki export file with mixed note type data:
      """
      Chinese	xue2	学	[sound:xue2.mp3]	to learn
      Chinese 2	空	kong1	[sound:kong1.mp3]	empty; sky
      """
    When I parse the file
    Then I should get 2 cards
    And card 1 should have notetype "Chinese" and characters "学"
    And card 2 should have notetype "Chinese 2" and characters "空"

  Scenario: Summary command should detect Chinese 2 cards
    Given I have an Anki export file "Chinese.txt" with Chinese 2 note type data:
      """
      Chinese 2	空	kong1	[sound:kong1.mp3]	empty; sky
      Chinese	xue2	学	[sound:xue2.mp3]	to learn
      """
    When I run the summary command on "Chinese.txt"
    Then the summary should show 2 total cards
    And the summary should show 2 unique characters