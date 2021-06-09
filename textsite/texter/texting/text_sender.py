from twilio.rest import Client

def send_messages(name_numbers : dict, text : str, link : bool, link_txt, signature : str,auth_token : str):

    account_sid = 'AC3b2542929073e9b44b685a0e22507bea'
    authentication_token = auth_token[0]
    print(authentication_token)
    client = Client(account_sid,authentication_token)

    counter = 1
    for name,numbers in name_numbers.items():

        for number in numbers:
            print(number)
            txt = text
            if link:
                txt += '\n' + link_txt
            
            txt += '\n' + signature

            message = client.messages.create(body= txt, from_=[+13252083966], to=[number])
            print("Message " + str(counter) + " sent to " + name)
            counter += 1
