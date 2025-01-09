import argparse
import sys
import os
import xml.etree.ElementTree as ET

sys.path.append('.')

from utils import split_train_test
from utils.file_utils import read_file, save_to_file
from utils.xml_utils import parse_xml_content


def main():
    parser = argparse.ArgumentParser(description='Parse XML files and optionally split into train/test datasets.')
    parser.add_argument('input_file', help='Input XML file')
    parser.add_argument('--split', action='store_true', help='Split the parsed data into train and test files')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for the test set (default: 0.2)')
    parser.add_argument('--train_file', default=None, help='Output file for training data (default: train_<input_file>.txt)')
    parser.add_argument('--test_file', default=None, help='Output file for testing data (default: test_<input_file>.txt)')
    parser.add_argument('--output_dir', default='.', help='Directory to save output files (default: current directory)')

    try:
        args = parser.parse_args()

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        file_name_without_extension = os.path.splitext(os.path.basename(args.input_file))[0]

        train_file = os.path.join(output_dir, args.train_file or f"train_{file_name_without_extension}.txt")
        test_file = os.path.join(output_dir, args.test_file or f"test_{file_name_without_extension}.txt")

        xml_content = read_file(args.input_file)
        parsed_data = parse_xml_content(xml_content)

        if args.split:
            input_text = "\n".join(parsed_data)
            train_data, test_data = split_train_test(
                input_text=input_text,
                test_size=args.test_size
            )
            save_to_file(train_data, train_file)
            save_to_file(test_data, test_file)

            print(f"Data has been split and saved:\n"
                  f" - Training Data: '{train_file}'\n"
                  f" - Testing Data: '{test_file}'")
        else:
            output_file = os.path.join(output_dir, f"{file_name_without_extension}_parsed.txt")
            save_to_file(parsed_data, output_file)
            print(f"Parsed data has been saved to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.")
        sys.exit(1)
    except ET.ParseError:
        print(f"Error: The file '{args.input_file}' contains invalid XML.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
