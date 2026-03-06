def validate_submission(submission):
    # Implement validation logic for the chess tournament submission here
    # Example checks might include:
    # 1. Ensure the submission format is correct
    # 2. Check for any disallowed moves or invalid formats
    # 3. Validate that the player IDs are recognized
    # This is a placeholder for your actual validation code
    valid = True  # Replace this with actual validation logic
    return valid

# Example usage:
if __name__ == '__main__':
    submission_data = {}  # Replace with actual submission data
    if validate_submission(submission_data):
        print('Submission is valid')
    else:
        print('Submission is invalid')