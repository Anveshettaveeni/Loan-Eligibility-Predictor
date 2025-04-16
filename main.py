import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class LoanEligibilityPredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.min_income = 3000  # Minimum monthly income threshold
        self.min_cibil = 650    # Minimum credit score threshold
        self.max_loan_to_income = 0.5  # Max loan amount to income ratio

    def load_data(self, filepath):
        """Load and preprocess the loan application data"""
        data = pd.read_csv(filepath)
        
        # Data cleaning
        data['Dependents'] = data['Dependents'].replace('3+', '3')
        data['Dependents'] = data['Dependents'].fillna('0')
        data['Dependents'] = data['Dependents'].astype(int)
        
        # Fill missing values
        for col in ['Gender', 'Married', 'Self_Employed', 'Credit_History']:
            data[col] = data[col].fillna(data[col].mode()[0])
            
        data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
        data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median())
        
        return data

    def preprocess_data(self, data):
        """Encode categorical variables and create features"""
        # Encode categorical variables
        categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 
                          'Property_Area', 'Credit_History']
        
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            self.encoders[col] = le
            
        # Create new features
        data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
        data['Loan_to_Income_Ratio'] = data['LoanAmount'] / data['Total_Income']
        data['EMI'] = data['LoanAmount'] / data['Loan_Amount_Term']
        
        return data

    def train_model(self, data):
        """Train the Random Forest classifier"""
        X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
        y = data['Loan_Status']
        
        # Encode target variable
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.encoders['Loan_Status'] = le
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
    def save_model(self, filepath):
        """Save the trained model and encoders"""
        joblib.dump({
            'model': self.model,
            'encoders': self.encoders,
            'min_income': self.min_income,
            'min_cibil': self.min_cibil,
            'max_loan_to_income': self.max_loan_to_income
        }, filepath)
        print(f"Model saved to {filepath}")
        
    def load_saved_model(self, filepath):
        """Load a previously saved model"""
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.encoders = saved_data['encoders']
        self.min_income = saved_data['min_income']
        self.min_cibil = saved_data['min_cibil']
        self.max_loan_to_income = saved_data['max_loan_to_income']
        print("Model loaded successfully")
        
    def predict_eligibility(self, applicant_data):
        """Predict loan eligibility for a new applicant"""
        # Convert applicant data to DataFrame
        df = pd.DataFrame([applicant_data])
        
        # Preprocess the data
        df['Dependents'] = df['Dependents'].replace('3+', '3').fillna('0').astype(int)
        
        # Encode categorical variables
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
                
        # Create derived features
        df['Total_Income'] = df['ApplicantIncome'] + df.get('CoapplicantIncome', 0)
        df['Loan_to_Income_Ratio'] = df['LoanAmount'] / df['Total_Income']
        df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
        
        # Check basic eligibility criteria
        basic_checks = self._check_basic_eligibility(applicant_data)
        if not basic_checks['eligible']:
            return basic_checks
        
        # Make prediction
        features = df.drop(['Loan_ID'], axis=1, errors='ignore')
        proba = self.model.predict_proba(features)[0]
        prediction = self.model.predict(features)[0]
        
        # Decode prediction
        status = self.encoders['Loan_Status'].inverse_transform([prediction])[0]
        
        return {
            'eligible': status == 'Y',
            'probability': float(proba[1]),  # Probability of approval
            'message': 'Processed with ML model',
            'basic_checks_passed': True,
            'model_prediction': status
        }
        
    def _check_basic_eligibility(self, applicant_data):
        """Check basic eligibility criteria before using ML model"""
        messages = []
        eligible = True
        
        # Check minimum income
        total_income = applicant_data['ApplicantIncome'] + applicant_data.get('CoapplicantIncome', 0)
        if total_income < self.min_income:
            messages.append(f"Insufficient income (minimum ${self.min_income} required)")
            eligible = False
            
        # Check credit history (CIBIL score)
        if applicant_data.get('Credit_History', 0) < self.min_cibil:
            messages.append(f"Low credit score (minimum {self.min_cibil} required)")
            eligible = False
            
        # Check loan to income ratio
        loan_to_income = applicant_data['LoanAmount'] / total_income
        if loan_to_income > self.max_loan_to_income:
            messages.append(f"Loan amount too high for income (max {self.max_loan_to_income*100:.0f}% of income)")
            eligible = False
            
        return {
            'eligible': eligible,
            'message': ' | '.join(messages) if messages else 'Passed basic checks',
            'basic_checks_passed': eligible,
            'model_prediction': None
        }

def train_new_model():
    """Train a new model from the dataset"""
    predictor = LoanEligibilityPredictor()
    data = predictor.load_data('loan_data.csv')
    processed_data = predictor.preprocess_data(data)
    predictor.train_model(processed_data)
    predictor.save_model('loan_predictor_model.joblib')
    return predictor

def load_existing_model():
    """Load an existing trained model"""
    predictor = LoanEligibilityPredictor()
    predictor.load_saved_model('loan_predictor_model.joblib')
    return predictor

def interactive_application():
    """Interactive loan application interface"""
    try:
        predictor = load_existing_model()
    except:
        print("No trained model found. Training a new model...")
        predictor = train_new_model()
        
    print("\nLoan Eligibility Prediction System")
    print("="*40)
    
    # Collect applicant information
    applicant_data = {
        'Loan_ID': 'LP001023',
        'Gender': input("Gender (Male/Female): ").capitalize(),
        'Married': input("Married? (Y/N): ").upper(),
        'Dependents': input("Number of dependents (0-3+): "),
        'Education': input("Education (Graduate/Under Graduate): ").capitalize(),
        'Self_Employed': input("Self employed? (Y/N): ").upper(),
        'ApplicantIncome': float(input("Applicant monthly income ($): ")),
        'CoapplicantIncome': float(input("Co-applicant monthly income ($) [0 if none]: ")),
        'LoanAmount': float(input("Loan amount requested ($): ")),
        'Loan_Amount_Term': float(input("Loan term (months): ")),
        'Credit_History': float(input("Credit (CIBIL) score (300-900): ")),
        'Property_Area': input("Property area (Urban/Semiurban/Rural): ").capitalize()
    }
    
    # Predict eligibility
    result = predictor.predict_eligibility(applicant_data)
    
    print("\nLoan Eligibility Result:")
    print("="*40)
    if result['eligible']:
        print("✅ APPROVED")
        print(f"Approval probability: {result['probability']*100:.1f}%")
    else:
        print("❌ REJECTED")
        if not result['basic_checks_passed']:
            print("Reason:", result['message'])
        else:
            print("Based on our risk assessment model")
    
    if result['basic_checks_passed']:
        print(f"\nModel prediction: {'Approved' if result['model_prediction'] == 'Y' else 'Rejected'}")
        print(f"Confidence: {result['probability']*100:.1f}%")

if __name__ == "__main__":
    interactive_application()
