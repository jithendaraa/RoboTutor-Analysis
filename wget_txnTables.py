import os
txn_txt_google_drive_links = ['https://drive.google.com/file/d/1SeMa3mlNnJgYyebUvFWhd9TDTXcOGcut/view?usp=sharing',
                                'https://drive.google.com/file/d/1ykoJrsyIsBrFrLqdj6DcLzMrBBzkqDXD/view?usp=sharing',
                                'https://drive.google.com/file/d/1YT0pZltOdkfZjYkKU3I4pjnb9ONv-_Re/view?usp=sharing',
                                'https://drive.google.com/file/d/1psRTMfS_4WYBH0Qo3V48kVcLldtb7WA1/view?usp=sharing',
                                'https://drive.google.com/file/d/1dKc3E3RXdzp4N2zoPyvkHCW-eASOBv5L/view?usp=sharing',
                                'https://drive.google.com/file/d/1MU3tfeQtClECkSSCPe22aDQQZ_41_15u/view?usp=sharing',
                                'https://drive.google.com/file/d/1Z7E-TxUhQZ2FcMeNTfh7tLQeS4jZwbG0/view?usp=sharing',
                                'https://drive.google.com/file/d/1pELCbp6wAjWgh04I6qr_kbGR_IWli-3F/view?usp=sharing',
                                'https://drive.google.com/file/d/1kdvo5Uczuad_Sf_z1oG3eR_oAL34XAxX/view?usp=sharing',
                                'https://drive.google.com/file/d/1Phy0n-iSSJukyXLXNWob-SNybL6SAGpM/view?usp=sharing',
                                'https://drive.google.com/file/d/1V8M4LDrKh_jogEKw06H9QOyl1XeWZ_IM/view?usp=sharing',
                                'https://drive.google.com/file/d/1JYpLoS56AliBJ-YnB4fWVR6zAkTnecns/view?usp=sharing',
                                'https://drive.google.com/file/d/1ylF1B670UtqUUEIrsPMsw6-MZfMuyAag/view?usp=sharing',
                                'https://drive.google.com/file/d/1eiOvKPALxQI-BJDTL9d5uRCj4csknjuL/view?usp=sharing',   #127
                                'https://drive.google.com/file/d/1rYTA52pM9Vw6-Fx6hC91KyNQiTEPN2FK/view?usp=sharing',
                                'https://drive.google.com/file/d/1CZFXCb-_qZA1FkRsBhJSwgaxqlBCvUrr/view?usp=sharing',
                                'https://drive.google.com/file/d/1uiYuDPrXi4_DkImG_Plj1WPN3XPxKQwp/view?usp=sharing',
                                'https://drive.google.com/file/d/1OSBJ9oHK4L_1apSHdVKC_GHzpkrl0h8z/view?usp=sharing',
                                'https://drive.google.com/file/d/1rtYuATHI4Nh8NPIvjeEq3ePGE9t-VSbp/view?usp=sharing',   #132
                                'https://drive.google.com/file/d/1VgwWcPekF3XsXITs5meYyZBs_uJNl3OX/view?usp=sharing',
                                'https://drive.google.com/file/d/1i5jPrP67-MbMeBMEd5Z3u_HtOV0zJ9KQ/view?usp=sharing',
                                'https://drive.google.com/file/d/1TknUTS85UXlhmQqJQQx-AtBIGtgAmV4o/view?usp=sharing',
                                'https://drive.google.com/file/d/1tiOEl0DmXNgxSkxtogHFjLuB0nKd7GIg/view?usp=sharing',
                                'https://drive.google.com/file/d/1R_dbAtca3g8VSvgarjPhpJkOVKv6CIEC/view?usp=sharing',
                                'https://drive.google.com/file/d/1nZThiW6Scfx5jBrmj3hbbAjogbkJTf5Y/view?usp=sharing',   # 138
                                'https://drive.google.com/file/d/1O6YBzFImodTON-Xlc2wNoTBMiTl1o0Vh/view?usp=sharing',
                                'https://drive.google.com/file/d/1VyO1M5bVnAVjuri8xml2CiZRbr2cXrn8/view?usp=sharing',
                                'https://drive.google.com/file/d/1l0puEgAIe0gauEK1QmDtmLiM1T0PjrEQ/view?usp=sharing',
                                ]

for i in range(len(txn_txt_google_drive_links[:1])):
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
    