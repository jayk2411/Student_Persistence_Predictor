async function predictPersistence() {
    const data = {
        first_term_gpa: parseFloat(document.getElementById('firstTermGpa').value),
        second_term_gpa: parseFloat(document.getElementById('secondTermGpa').value),
        high_school_average: parseFloat(document.getElementById('highSchoolAverage').value),
        math_score: parseFloat(document.getElementById('mathScore').value)
    };

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        
        document.getElementById('result').style.display = 'block';
        document.getElementById('predictionText').innerText = 
            `Probability of Persistence: ${(result.probability * 100).toFixed(1)}%`;
        document.getElementById('probabilityFill').style.width = 
            `${result.probability * 100}%`;

        // Update history
        loadPredictionHistory();
    } catch (error) {
        console.error('Error:', error);
        alert('Error making prediction. Please try again.');
    }
}

async function loadPredictionHistory() {
    try {
        const response = await fetch('http://localhost:5000/history');
        const history = await response.json();
        
        const tbody = document.querySelector('#historyTable tbody');
        tbody.innerHTML = '';
        
        history.forEach(record => {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = new Date(record.timestamp).toLocaleString();
            row.insertCell(1).textContent = record.input.first_term_gpa;
            row.insertCell(2).textContent = record.input.second_term_gpa;
            row.insertCell(3).textContent = record.input.high_school_average;
            row.insertCell(4).textContent = record.input.math_score;
            row.insertCell(5).textContent = `${(record.prediction * 100).toFixed(1)}%`;
        });
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Load history when page loads
document.addEventListener('DOMContentLoaded', loadPredictionHistory);
