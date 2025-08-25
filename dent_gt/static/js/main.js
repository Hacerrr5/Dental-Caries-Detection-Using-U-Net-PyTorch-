// Smooth scroll navigation
document.querySelectorAll('nav a').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
    link.classList.add('active');
    const target = document.querySelector(link.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth' });
  });
});

// Appointment form submission
const appointmentForm = document.getElementById('appointmentForm');
appointmentForm.addEventListener('submit', e => {
  e.preventDefault();

  if (!appointmentForm.checkValidity()) {
    alert('Please fill in all required fields correctly.');
    return;
  }

  alert('Thank you! Your appointment request has been received. We will contact you shortly.');
  appointmentForm.reset();
});

// Review form logic
const reviewForm = document.getElementById('reviewForm');
const reviewsGrid = document.getElementById('reviewsGrid');

reviewForm.addEventListener('submit', e => {
  e.preventDefault();

  // Basic validation
  if (!reviewForm.checkValidity()) {
    alert('Please fill in all required fields correctly.');
    return;
  }

  // Get values
  const name = document.getElementById('reviewerName').value.trim();
  const rating = document.getElementById('reviewRating').value;
  const message = document.getElementById('reviewMessage').value.trim();

  // Generate stars string
  const starsFilled = '★'.repeat(rating);
  const starsEmpty = '☆'.repeat(5 - rating);
  const starsDisplay = starsFilled + starsEmpty;

  // Create review card element
  const reviewCard = document.createElement('article');
  reviewCard.classList.add('review-card');
  reviewCard.setAttribute('tabindex', '0');
  reviewCard.innerHTML = `
    <div class="review-author">${name}</div>
    <div class="stars" aria-label="${rating} out of 5 stars">${starsDisplay}</div>
    <div class="review-text">${message}</div>
  `;

  // Append to grid and reset form
  reviewsGrid.appendChild(reviewCard);
  reviewForm.reset();

  alert('Thank you for your valuable review!');
});

// --- Machine Learning Section Logic ---
const mlImageUpload = document.getElementById('ml-image-upload');
const mlPreview = document.getElementById('ml-preview');
const mlResult = document.getElementById('ml-result');
let selectedMLImage = null; // Stores the uploaded file object

// Show preview when file is selected
mlImageUpload.addEventListener('change', function () {
  const file = this.files[0];
  selectedMLImage = file; // Save the selected file
  mlPreview.innerHTML = ''; // Clear previous preview
  mlResult.textContent = ''; // Clear previous result

  if (file && file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = function (e) {
      const img = document.createElement('img');
      img.src = e.target.result;
      img.alt = "Preview of uploaded dental image";
      img.className = "ml-preview-img"; // Use existing CSS styling
      mlPreview.appendChild(img);
    };
    reader.readAsDataURL(file);
  } else {
    mlResult.textContent = "Please select a valid image file (.jpg, .png, etc.).";
  }
});

// Run ML prediction when "Do Caries Analysis" button is clicked
function runMLDetection() {
  mlResult.textContent = ''; // Clear previous text result

  if (!selectedMLImage) {
    mlResult.textContent = "Please upload an image before running the analysis.";
    return;
  }

  mlResult.textContent = "Analysis in progress..."; // Show loading message

  // Create FormData object to send file to server
  const formData = new FormData();
  // 'image' is the key Flask will look for in request.files['image']
  formData.append('image', selectedMLImage);

  // Send POST request to Flask backend
  fetch('/predict', {
    method: 'POST',
    body: formData
  })
    .then(response => {
      // Check response status
      if (!response.ok) {
        return response.text().then(text => {
          throw new Error(`HTTP Error! Status: ${response.status}, Response: ${text}`);
        });
      }
      return response.json(); // Parse JSON response
    })
    .then(data => {
      if (data.success) {
        // Show base64 encoded annotated image
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + data.image_data; // Add prefix for base64 data
        img.alt = "Analyzed Image";
        // Apply CSS styling for better display
        img.style.width = '100%';
        img.style.height = 'auto';
        img.style.maxWidth = '600px';
        img.style.borderRadius = '10px';
        img.style.marginTop = '0.9rem';
        img.style.border = '2px solid #b8860b';
        img.style.boxShadow = '0 6px 20px #b8860b1a';

        mlPreview.innerHTML = ''; // Clear old preview before showing new one
        mlPreview.appendChild(img);

        // Display prediction details
        mlResult.innerHTML = data.predictions.join('<br>');
      } else {
        // success: false from Flask backend (application error)
        mlResult.textContent = "Error: " + (data.error || "An unknown error occurred.");
      }
    })
    .catch(error => {
      // Handle network or parsing errors
      console.error('Fetch error (Browser Console):', error);
      mlResult.textContent =
        "An error occurred while communicating with the server. Please check Developer Console (F12). Details: " +
        error.message;
    });
}
