class BankAccount:
    """A class representing a bank account with basic operations."""
    
    # Class variable (shared across all instances)
    bank_name = "Python Bank"
    account_count = 0
    
    def __init__(self, owner, balance=0):
        """Initialize a new bank account.
        
        Args:
            owner (str): Name of the account owner
            balance (float): Initial balance (default 0)
        """
        self.owner = owner
        self._balance = balance  # Protected attribute
        self.transaction_history = []
        self.account_number = BankAccount.account_count + 1
        BankAccount.account_count += 1
        self._log_transaction("Account opened", balance)
    
    def deposit(self, amount):
        """Deposit money into the account.
        
        Args:
            amount (float): Amount to deposit
            
        Returns:
            bool: True if successful, False otherwise
        """
        if amount <= 0:
            print("Deposit amount must be positive")
            return False
        
        self._balance += amount
        self._log_transaction("Deposit", amount)
        return True
    
    def withdraw(self, amount):
        """Withdraw money from the account.
        
        Args:
            amount (float): Amount to withdraw
            
        Returns:
            bool: True if successful, False otherwise
        """
        if amount <= 0:
            print("Withdrawal amount must be positive")
            return False
        
        if amount > self._balance:
            print("Insufficient funds")
            return False
        
        self._balance -= amount
        self._log_transaction("Withdrawal", -amount)
        return True
    
    def get_balance(self):
        """Return the current balance."""
        return self._balance
    
    def _log_transaction(self, transaction_type, amount):
        """Private method to log transactions.
        
        Args:
            transaction_type (str): Type of transaction
            amount (float): Transaction amount
        """
        self.transaction_history.append({
            'type': transaction_type,
            'amount': amount,
            'balance': self._balance
        })
    
    def print_statement(self):
        """Print account statement with transaction history."""
        print(f"\n{'='*50}")
        print(f"{self.bank_name} - Account Statement")
        print(f"{'='*50}")
        print(f"Account Number: {self.account_number}")
        print(f"Owner: {self.owner}")
        print(f"Current Balance: ${self._balance:.2f}")
        print(f"\nTransaction History:")
        print(f"{'-'*50}")
        
        for i, transaction in enumerate(self.transaction_history, 1):
            print(f"{i}. {transaction['type']}: ${transaction['amount']:.2f} "
                  f"(Balance: ${transaction['balance']:.2f})")
        print(f"{'='*50}\n")
    
    @classmethod
    def get_bank_info(cls):
        """Class method to get bank information."""
        return f"{cls.bank_name} - Total Accounts: {cls.account_count}"
    
    @staticmethod
    def validate_account_number(account_num):
        """Static method to validate account number format."""
        return isinstance(account_num, int) and account_num > 0
    
    def __str__(self):
        """String representation of the account."""
        return f"BankAccount({self.owner}, ${self._balance:.2f})"
    
    def __repr__(self):
        """Developer-friendly representation."""
        return f"BankAccount(owner='{self.owner}', balance={self._balance})"
    
    def __eq__(self, other):
        """Check equality based on account number."""
        if isinstance(other, BankAccount):
            return self.account_number == other.account_number
        return False


# Example usage
if __name__ == "__main__":
    # Create accounts
    account1 = BankAccount("Alice", 1000)
    account2 = BankAccount("Bob", 500)
    
    # Perform operations
    account1.deposit(500)
    account1.withdraw(200)
    account1.deposit(100)
    
    account2.deposit(300)
    account2.withdraw(50)
    
    # Print statements
    account1.print_statement()
    account2.print_statement()
    
    # Use class method
    print(BankAccount.get_bank_info())
    
    # Use static method
    print(f"Is account number valid? {BankAccount.validate_account_number(1)}")
    
    # String representation
    print(account1)
    print(repr(account2))