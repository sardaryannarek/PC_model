def read_file(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def save_to_file(content, output_filename):

    with open(output_filename, 'w', encoding='utf-8') as file:
        for item in content:
            file.write(f"{item}\n")
