from main import check_email, update_checked_item_list, get_xxh3_128, is_checked, get_checked_item_list

if __name__ == '__main__':
    update_mail_list = check_email()
    for each_email in update_mail_list:
        print(each_email['subject'])
        email_hash = f'{each_email["subject"]}\n{each_email["date"]}\n{each_email["body"]}'
        email_hash = get_xxh3_128(email_hash)
        if is_checked(email_hash):
            print(f'email already checked: {each_email["subject"]}, hash: {email_hash}')
            continue
        update_checked_item_list(email_hash, each_email['subject'])
    
    print('all emails checked and updated.')
    result = get_checked_item_list()
    for item in result:
        print(item)

        