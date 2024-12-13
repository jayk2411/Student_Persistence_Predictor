async function predictPersistence() {
    // Validate inputs first
    const inputs = {
        firstTermGpa: document.getElementById('firstTermGpa').value,
        secondTermGpa: document.getElementById('secondTermGpa').value,
        highSchoolAverage: document.getElementById('highSchoolAverage').value,
        mathScore: document.getElementById('mathScore').value
    };

    // Check if any input is empty
    for (const [key, value] of Object.entries(inputs)) {
        if (!value) {
            alert('Please fill in all fields');
            return;
        }
    }

    const data = {
        first_term_gpa: parseFloat(inputs.firstTermGpa),
        second_term_gpa: parseFloat(inputs.secondTermGpa),
        high_school_average: parseFloat(inputs.highSchoolAverage),
        math_score: parseFloat(inputs.mathScore)
    };

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        document.getElementById('result').style.display = 'block';
        document.getElementById('predictionText').innerText = 
            `Probability of Persistence: ${(result.probability * 100).toFixed(1)}%`;
        document.getElementById('probabilityFill').style.width = 
            `${result.probability * 100}%`;

        // Update history
        await loadPredictionHistory();
    } catch (error) {
        console.error('Error:', error);
        handleServerError('Error making prediction. Please ensure the server is running.');
    }
}

async function loadPredictionHistory() {
    try {
        const response = await fetch('http://localhost:5000/history');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const history = await response.json();
        
        const tbody = document.querySelector('#historyTable tbody');
        tbody.innerHTML = '';
        
        if (history.length === 0) {
            // Show message when no history exists
            const row = tbody.insertRow();
            const cell = row.insertCell(0);
            cell.colSpan = 6;
            cell.textContent = 'No prediction history available';
            cell.style.textAlign = 'center';
            cell.style.padding = '20px';
        } else {
            history.forEach(record => {
                const row = tbody.insertRow();
                row.insertCell(0).textContent = new Date(record.timestamp).toLocaleString();
                row.insertCell(1).textContent = record.input.first_term_gpa;
                row.insertCell(2).textContent = record.input.second_term_gpa;
                row.insertCell(3).textContent = record.input.high_school_average;
                row.insertCell(4).textContent = record.input.math_score;
                row.insertCell(5).textContent = `${(record.prediction * 100).toFixed(1)}%`;
            });
        }

        // Show the history section
        document.getElementById('history').style.display = 'block';

    } catch (error) {
        console.error('Error loading history:', error);
        handleServerError('Unable to load prediction history. Please ensure the server is running.');
    }
}

function handleServerError(message) {
    // Create or update error message
    let errorDiv = document.getElementById('serverError');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.id = 'serverError';
        errorDiv.style.cssText = `
            background-color: #ffebee;
            color: #c62828;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            text-align: center;
        `;
        document.querySelector('.container').insertBefore(
            errorDiv, 
            document.querySelector('.input-form')
        );
    }
    errorDiv.textContent = message;

    // Hide the history section if there's an error
    document.getElementById('history').style.display = 'none';
}

function initializeApp() {
    // Hide the result section initially
    document.getElementById('result').style.display = 'none';
    
    // Try to load prediction history
    loadPredictionHistory().catch(error => {
        console.error('Initial history load failed:', error);
        handleServerError('Server connection failed. Please ensure the server is running.');
    });
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', initializeApp);