"""
Loan Risk Analytics API
Serves synthetic customer loan data for the risk analytics dashboard
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Loan Risk Analytics API",
    description="API for customer loan data used in risk prediction",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerLoanData(BaseModel):
    customer_id: int
    age: int
    monthly_income: float
    loan_amount: float
    credit_score: int
    employment_years: int
    defaulted: int  # 0 = No, 1 = Yes


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer loan data with realistic patterns.
    Higher default rates for: lower credit scores, higher loan amounts relative to income,
    shorter employment history, and younger age.
    """
    np.random.seed(seed)
    
    # Generate base features
    customer_ids = list(range(1, n_samples + 1))
    ages = np.random.randint(21, 65, n_samples)
    monthly_incomes = np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples)
    monthly_incomes = np.clip(monthly_incomes, 15000, 500000).astype(int)
    
    # Loan amounts - correlated with income but with variation
    loan_amounts = monthly_incomes * np.random.uniform(3, 24, n_samples)
    loan_amounts = np.clip(loan_amounts, 50000, 5000000).astype(int)
    
    # Credit scores - normally distributed
    credit_scores = np.random.normal(680, 80, n_samples)
    credit_scores = np.clip(credit_scores, 300, 850).astype(int)
    
    # Employment years - correlated with age
    max_employment = np.maximum(ages - 21, 1)
    employment_years = np.array([
        np.random.randint(0, min(max_emp, 30) + 1) 
        for max_emp in max_employment
    ])
    
    # Calculate default probability based on risk factors
    # Higher probability for: low credit score, high debt-to-income, low employment, young age
    debt_to_income = loan_amounts / (monthly_incomes * 12)
    
    # Normalize factors to 0-1 scale for probability calculation
    credit_risk = (850 - credit_scores) / 550  # Higher for lower scores
    dti_risk = np.clip(debt_to_income / 3, 0, 1)  # Higher for higher DTI
    employment_risk = np.clip((10 - employment_years) / 10, 0, 1)  # Higher for less experience
    age_risk = np.clip((35 - ages) / 14, 0, 1)  # Higher for younger
    
    # Combined risk score
    risk_score = (
        0.35 * credit_risk + 
        0.30 * dti_risk + 
        0.20 * employment_risk + 
        0.15 * age_risk
    )
    
    # Add some noise and convert to binary outcome
    risk_score = np.clip(risk_score + np.random.normal(0, 0.1, n_samples), 0, 1)
    default_threshold = np.random.uniform(0, 1, n_samples)
    defaulted = (risk_score > 0.55) & (default_threshold < risk_score)
    defaulted = defaulted.astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'monthly_income': monthly_incomes,
        'loan_amount': loan_amounts,
        'credit_score': credit_scores,
        'employment_years': employment_years,
        'defaulted': defaulted
    })
    
    return df


# Generate data at startup
loan_data = generate_synthetic_data(n_samples=1000)


@app.get("/")
def root():
    """API root endpoint with documentation"""
    return {
        "message": "Loan Risk Analytics API",
        "endpoints": {
            "/customers": "Get all customer loan data",
            "/customers/{customer_id}": "Get specific customer data",
            "/customers/sample": "Get a sample of customer data",
            "/stats": "Get summary statistics"
        }
    }


@app.get("/customers", response_model=List[CustomerLoanData])
def get_all_customers():
    """Get all customer loan data"""
    return loan_data.to_dict(orient='records')


@app.get("/customers/sample", response_model=List[CustomerLoanData])
def get_sample_customers(n: int = Query(default=100, ge=1, le=1000)):
    """Get a random sample of customer data"""
    sample = loan_data.sample(n=min(n, len(loan_data)))
    return sample.to_dict(orient='records')


@app.get("/customers/{customer_id}", response_model=CustomerLoanData)
def get_customer(customer_id: int):
    """Get data for a specific customer"""
    customer = loan_data[loan_data['customer_id'] == customer_id]
    if customer.empty:
        return {"error": "Customer not found"}
    return customer.iloc[0].to_dict()


@app.get("/stats")
def get_statistics():
    """Get summary statistics of the loan portfolio"""
    stats = {
        "total_customers": len(loan_data),
        "default_rate": float(loan_data['defaulted'].mean()),
        "total_defaults": int(loan_data['defaulted'].sum()),
        "avg_credit_score": float(loan_data['credit_score'].mean()),
        "avg_loan_amount": float(loan_data['loan_amount'].mean()),
        "avg_monthly_income": float(loan_data['monthly_income'].mean()),
        "avg_age": float(loan_data['age'].mean()),
        "avg_employment_years": float(loan_data['employment_years'].mean())
    }
    return stats


@app.post("/refresh")
def refresh_data(n_samples: int = 1000, seed: Optional[int] = None):
    """Regenerate synthetic data with optional new seed"""
    global loan_data
    if seed is None:
        seed = np.random.randint(0, 10000)
    loan_data = generate_synthetic_data(n_samples=n_samples, seed=seed)
    return {"message": f"Data refreshed with {n_samples} samples", "seed": seed}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

