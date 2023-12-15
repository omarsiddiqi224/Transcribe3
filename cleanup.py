import json

# Opening JSON file
with open('output.json') as f:
    data = json.load(f)

# Variable to keep track of the current speaker and their accumulated text
current_speaker = None
accumulated_text = ""

# Opening the file to write
with open('audio.txt', 'w') as output_file:
    for element in data['speakers']:
        speaker = element['speaker']
        text = element['text'].strip()  # Strip leading and trailing spaces

        # Check if the current element's speaker is the same as the last one
        if speaker == current_speaker:
            # Append the text with a single space
            accumulated_text += " " + text
        else:
            # If the speaker has changed and there is accumulated text, write it to the file
            if accumulated_text:
                output_file.write(f"{current_speaker}: {accumulated_text}\n\n")
            
            # Update the current speaker and reset the accumulated text
            current_speaker = speaker
            accumulated_text = text

    # Write the last accumulated text if there is any
    if accumulated_text:
        output_file.write(f"{current_speaker}: {accumulated_text}\n\n")

# Closing file
f.close()
