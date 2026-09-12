import os
import time
import ast
from datetime import datetime

class RentalService:
    
    def __init__(self):
        self.cust_data = {}
        self.car_data = {}

    def keyboardInput(self,datatype, caption, errorMessage, defaultValue=None):
        value = None
        isInvalid = True
        while isInvalid:
            try:
                if defaultValue is None:
                    value = datatype(input(caption))
                else:
                    value = input(caption)
                    if value.strip() == "":
                        value = defaultValue
                    else:
                        value = datatype(value)
            except:
                print(errorMessage)
            else:
                isInvalid = False
        return value

    def clear_screen(self):
        # Clear screen based on operating system
        os.system('cls' if os.name == 'nt' else 'clear')


    # Read from txt files #####################################################################################################################################


    def read_data(self):
        # global cust_data, car_data
        try:
            with open('customerdetails.txt', "rt") as cust_detail:
                self.cust_data = ast.literal_eval(cust_detail.read())

            with open('carlisting.txt', "rt") as car_listing:
                self.car_data = ast.literal_eval(car_listing.read())

        except Exception as e:
            print("Something went wrong when we print the data:", e)

    def getcarID(self,customer_id):
        # global cust_data
        if customer_id in self.cust_data:
            details = self.cust_data[customer_id]
            return details['CarID']
        else:
            print("Invalid Customer ID")
            return None
        
    def read_customer_id(self):
        # global cust_data
        
        while True:
            customer_id = input("Enter customer ID, etc: 001: ").strip()
            if customer_id in self.cust_data:
                return customer_id
            else:
                print("ID Not Found")


    # Calculation ##############################################################################################################################################


    def calculate_rental_days(self,start_date, end_date):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        rental_days = (end_date - start_date).days + 1
        return rental_days

    def calculate_payment(self,customer_id):
        # global cust_data, car_data

        if customer_id:
            customer = self.cust_data[customer_id]
            car_id = customer.get('CarID')
            car = self.car_data[car_id]

            start_date = datetime.strptime(customer['Startdate'], '%Y-%m-%d')
            end_date = datetime.strptime(customer['Enddate'], '%Y-%m-%d')
            price_per_day = float(car['Price/day'].replace('RM', ''))
            balance = float(customer['Balance'].replace('RM', ''))
            days_of_rent = (end_date - start_date).days + 1
            total_payment = days_of_rent * price_per_day

            if balance < total_payment:
                print("Your balance is insufficient. Choose another payment method.")
                self.digital_wallet(customer_id)
            else:
                new_balance = balance - total_payment
                # clear_screen()
                time.sleep(2)
                print("Your payment is processing....")
                time.sleep(3)
                self.clear_screen()
                time.sleep(2)
                print("Your payment is successful")
                print(f"Your new account balance: RM {new_balance:.2f}")
                if new_balance is not None:
                    self.cust_data[customer_id]['Balance'] = f"RM {new_balance:.2f}"
                    with open('updated_customerdetails.txt', 'w') as file:
                        file.write(str(self.cust_data))
                    self.print_invoice(customer_id, self.payment_method)
                return new_balance


    # Payment Details ##########################################################################################################################################


    def payment_method(self,customer_id):
        while True:
            print("-------------------------")
            print("| 1 - Credit Card       |")
            print("| 2 - Digital Wallet    |")
            print("| 3 - Bank Transfer     |")
            print("| 4 - Cash Payment      |")
            print("| 5 - Exit              |")
            print("-------------------------")
            choice = self.keyboardInput(int, "Choice [1, 2, 3, 4, 5]: ", "Choice must be Integer")
            if choice == 1:
                self.credit_card(customer_id)
                break
            elif choice == 2:
                self.digital_wallet(customer_id)
                break
            elif choice == 3:
                self.bank_transfer(customer_id)
                break
            elif choice == 4:
                self.cash_payment(customer_id)
                break
            elif choice == 5:
                break
            else:
                print("Please choose your choice within range 1-5 only")

    def credit_card(self,customer_id):
        while True:
            credit_card_no = self.keyboardInput(str, "Enter the credit card number (16-Digits): ", "Credit Card Number must be an integer").replace(" ", "").strip()
            if len(credit_card_no) != 16 or not credit_card_no.isdigit():
                print("Invalid credit card")
                continue
            name = input("Enter Card Name Holder: ")
            try:
                while True:
                    expired_date = self.keyboardInput(str, "Enter credit card expired date in format MMYY: ", "Expired date must be an integer").replace("/", "").strip()
                    if len(expired_date) != 4 or not expired_date.isdigit():
                        print("Expired date must be in format MMYY")
                        continue
                    
                    month = int(expired_date[:2])
                    year = int(expired_date[2:])
                    current_year = datetime.now().year % 100

                    if month < 1 or month > 12:
                        print("Invalid month in expiry date")
                        continue
                    if year < current_year or (year == current_year and month < datetime.now().month):
                        print("Credit card has expired")
                        continue

                    while True:
                        cvv = self.keyboardInput(str, "Enter CVV number(3-Digits): ", "CVV must be an integer").strip()
                        if len(cvv) != 3 or not cvv.isdigit() or int(cvv) < 100:
                            print("Invalid CVV number")
                        else:
                            break
                    break      
            except ValueError:
                return "Invalid expiration date format"
            self.clear_screen()
            time.sleep(2)
            print("Your payment is processing...")
            time.sleep(3)
            self.clear_screen()
            time.sleep(2)
            print("Your payment is successful")
            self.print_invoice(customer_id, self.payment_method)
            # update_balance(customer_id, "Credit Card (PAID)")
            break

    def digital_wallet(self,customer_id):
        while True:
            print("----------------------")
            print("| 1 - Shopee Pay     |")
            print("| 2 - Grab Pay       |")
            print("| 3 - TnG            |")
            print("| 4 - Apple Pay      |")
            print("| 5 - Paypal         |")
            print("| 6 - Main Menu      |")
            print("----------------------")
            choice = self.keyboardInput(int, "Choice [1, 2, 3, 4, 5, 6]: ", "Choice must be Integer")
            if choice in range(1, 6):
                self.calculate_payment(customer_id)
                # update_balance(customer_id, "Digital Wallet (PAID)")
                break
            elif choice == 6:
                self.payment_method(customer_id)
                break
            else:
                print("Please choose your choice within range 1-6 only")

    def username_password(self,customer_id):
        username = input("Enter username: ")
        password = input("Enter password: ")
        print("\n\n")
        print("=============================================")
        print("Username:", username)
        print("Password:", '*' * len(password))
        self.calculate_payment(customer_id)

    def bank_transfer(self,customer_id):
        while True:
            print("-----------------------")
            print("| 1 - Maybank         |")
            print("| 2 - CIMB            |")
            print("| 3 - Bank Islam      |")
            print("| 4 - RHB             |")
            print("| 5 - BSN             |")
            print("| 6 - Main Menu       |")
            print("-----------------------")
            choice = self.keyboardInput(int, "Choice [1, 2, 3, 4, 5, 6]: ", "Choice must be Integer")
            if choice in range(1, 6):
                self.username_password(customer_id)
                # update_balance(customer_id, "Bank Transfer (PAID)")
                break
            elif choice == 6:
                self.payment_method(customer_id)
                break
            else:
                print("Please choose your choice within range 1-6 only")

    def cash_payment(self,customer_id):
        # global cust_data

        customer = self.cust_data[customer_id]
        car_id = customer.get('CarID')
        car = self.car_data[car_id]

        start_date = customer['Startdate']
        end_date = customer['Enddate']
        rental_days = self.calculate_rental_days(start_date, end_date)
        price_per_day = float(car['Price/day'].replace('RM', ''))
        total_rental_payment = rental_days * price_per_day

        deposit = total_rental_payment // 3  # Calculate deposit

        if customer_id:
            customer = self.cust_data[customer_id]
            self.clear_screen()
            time.sleep(2)
            print(f"Your payment is pending. Make sure to pay at the counter.")
            print("=" * 100)
            title = "SOCAR"
            print(f"{title:^100}")
            title = "RENTAL INVOICE"
            print(f"{title:^100}")
            print("=" * 100)
            title = "CUSTOMER DETAILS"
            print(f"{title:^100}")
            print("=" * 100)
            print(f"{'Name':<20}: {customer['Name']}")
            print(f"{'Address':<20}: {customer['Address']}")
            print(f"{'Phone':<20}: {customer['Phone']}")
            print(f"{'IC No':<20}: {customer['IC No']}")
            print(f"{'Startdate':<20}: {customer['Startdate']}")
            print(f"{'Enddate':<20}: {customer['Enddate']}")
            print("=" * 100)
            title = "RENTAL DETAILS"
            print(f"{title:^100}")
            print("=" * 100)
            print(f"{'Brand':<20}: {car['Brand']}")
            print(f"{'Type':<20}: {car['Type']}")
            print(f"{'Plate Number':<20}: {car['Plate Num']}")
            print(f"{'Price/day':<20}: {car['Price/day']}")
            print(f"{'Days rent':<20}: {rental_days}")
            print("=" * 100)
            print(f"{'Status':<20}:{' ':71}UNPAID")
            print(f"{'Grand Total':<20}:{' ':67} RM {total_rental_payment:.2f}")
            print(f"{'Deposit':<20}:{' ':67} RM {deposit:.2f}")
            print("=" * 100)


    # Main ####################################################################################################################################################


    def customerdetails(self,customer_id):
        # global cust_data
        if customer_id in self.cust_data:
            details = self.cust_data[customer_id]

            print("=" * 100)
            title = "SOCAR"
            print(f"{title:^100}")
            title = "RENTAL INVOICE"
            print(f"{title:^100}")
            print("=" * 100)
            title = "CUSTOMER DETAILS"
            print(f"{title:^100}")
            print("=" * 100)
            print(f"{'Name':<20}: {details['Name']}")
            print(f"{'Address':<20}: {details['Address']}")
            print(f"{'Phone':<20}: {details['Phone']}")
            print(f"{'IC No':<20}: {details['IC No']}")
            print(f"{'Startdate':<20}: {details['Startdate']}")
            print(f"{'Enddate':<20}: {details['Enddate']}")
            print("=" * 100)

            return details['Startdate'], details['Enddate']
        else:
            print("Invalid Customer ID")
            return None, None

    def cardetails(self,car_id, rental_days):
        # global car_data
        if car_id in self.car_data:
            details = self.car_data[car_id]

            title = "RENTAL DETAILS"
            print(f"{title:^100}")
            print("=" * 100)
            print(f"{'Brand':<20}: {details['Brand']}")
            print(f"{'Type':<20}: {details['Type']}")
            print(f"{'Plate Number':<20}: {details['Plate Num']}")
            print(f"{'Price/day':<20}: {details['Price/day']}")
            print(f"{'Days rent':<20}: {rental_days}")
            print("=" * 100)

            price_per_day_str = details['Price/day'].replace('RM', '').strip()
            price_per_day = float(price_per_day_str)
            total_rental_payment = price_per_day * rental_days

            deposit = total_rental_payment / 3  # Calculate deposit

            print(f"{'Grand Total':<20}:{' ':69} RM {total_rental_payment:.2f}")
            print(f"{'Deposit':<20}:{' ':70} RM {deposit:.2f}")
            print("=" * 100)
            return total_rental_payment

        else:
            print("Invalid Car ID")
            return None

    def print_invoice(self,customer_id, payment_method):
        # global cust_data, car_data

        customer = self.cust_data[customer_id]
        car_id = customer.get('CarID')
        car = self.car_data[car_id]

        start_date = customer['Startdate']
        end_date = customer['Enddate']
        rental_days = self.calculate_rental_days(start_date, end_date)
        price_per_day = float(car['Price/day'].replace('RM', ''))
        total_rental_payment = rental_days * price_per_day

        deposit = total_rental_payment // 3  # Calculate deposit

        print("=" * 100)
        title = "SOCAR"
        print(f"{title:^100}")
        title = "RENTAL INVOICE"
        print(f"{title:^100}")
        print("=" * 100)
        title = "CUSTOMER DETAILS"
        print(f"{title:^100}")
        print("=" * 100)
        print(f"{'Name':<20}: {customer['Name']}")
        print(f"{'Address':<20}: {customer['Address']}")
        print(f"{'Phone':<20}: {customer['Phone']}")
        print(f"{'IC No':<20}: {customer['IC No']}")
        print(f"{'Startdate':<20}: {customer['Startdate']}")
        print(f"{'Enddate':<20}: {customer['Enddate']}")
        print("=" * 100)
        title = "RENTAL DETAILS"
        print(f"{title:^100}")
        print("=" * 100)
        print(f"{'Brand':<20}: {car['Brand']}")
        print(f"{'Type':<20}: {car['Type']}")
        print(f"{'Plate Number':<20}: {car['Plate Num']}")
        print(f"{'Price/day':<20}: {car['Price/day']}")
        print(f"{'Days rent':<20}: {rental_days}")
        print("=" * 100)
        print(f"{'Status':<20}:{' ':71}PAID")
        print(f"{'Grand Total':<20}:{' ':67} RM {total_rental_payment:.2f}")
        print(f"{'Deposit':<20}:{' ':67} RM {deposit:.2f}")
        print("=" * 100)

    def main(self):
        self.read_data()
        customer_id = self.read_customer_id()
        if customer_id:
            start_date, end_date = self.customerdetails(customer_id)
            if start_date and end_date:
                car_id = self.getcarID(customer_id)
                rental_days = self.calculate_rental_days(start_date, end_date)
                self.cardetails(car_id, rental_days)
                self.payment_method(customer_id)
            else:
                print("Unable to retrieve start date and end date for the rental period.")
        else:
            print("Failed to read customer data")

if __name__ == "__main__":
    service = RentalService()
    service.main()