def generate_submission(predictions, output_file):
    with open(output_file, 'w') as f:
        print('row_id,open_flag', file=f)
        for i, pred in enumerate(predictions):
            print(f'{i},{pred}', file=f)
