import os
txn_txt_google_drive_links = ['https://drive.google.com/file/d/1SeMa3mlNnJgYyebUvFWhd9TDTXcOGcut/view?usp=sharing',
                                'https://drive.google.com/file/d/1ykoJrsyIsBrFrLqdj6DcLzMrBBzkqDXD/view?usp=sharing',
                                'https://drive.google.com/file/d/1YT0pZltOdkfZjYkKU3I4pjnb9ONv-_Re/view?usp=sharing',
                                'https://drive.google.com/file/d/1psRTMfS_4WYBH0Qo3V48kVcLldtb7WA1/view?usp=sharing',
                                'https://drive.google.com/file/d/1dKc3E3RXdzp4N2zoPyvkHCW-eASOBv5L/view?usp=sharing',
                                'https://drive.google.com/file/d/1MU3tfeQtClECkSSCPe22aDQQZ_41_15u/view?usp=sharing',
                                'https://drive.google.com/file/d/1Z7E-TxUhQZ2FcMeNTfh7tLQeS4jZwbG0/view?usp=sharing'
                                ]

for i in range(len(txn_txt_google_drive_links)):
    link = txn_txt_google_drive_links[i]
    village_num = 114 + i
    file_id = link.split('/')[-2:-1][0]
    file_name = "village_" + str(village_num) + "_KCSubtests.txt"
    print(file_id)
    os.chdir('Data')
    if "village_" + str(village_num) not in os.listdir('.'):
        os.system("mkdir village_" + str(village_num))
    os.chdir('village_' + str(village_num))
    wget_command = "wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=" + file_id + "' -O " + file_name
    os.system(wget_command)
    os.chdir('../../')
    print(wget_command)
    