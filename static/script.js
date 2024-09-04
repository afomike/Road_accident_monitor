function convertToMinutes(inputId) {
    const inputElement = document.getElementById(inputId);
    const inputValue = inputElement.value.trim();
    let totalMinutes = 0;
    
    // Regex patterns to match different possible formats
    const hourMinutePattern = /(\d+)\s*HOUR(?:S)?\s*(\d+)\s*MINUTE(?:S)?/i;
    const hourOnlyPattern = /(\d+)\s*HOUR(?:S)?/i;
    const minuteOnlyPattern = /(\d+)\s*MINUTE(?:S)?/i;

    if (hourMinutePattern.test(inputValue)) {
        const match = inputValue.match(hourMinutePattern);
        const hours = parseInt(match[1]);
        const minutes = parseInt(match[2]);
        totalMinutes = (hours * 60) + minutes;
    } else if (hourOnlyPattern.test(inputValue)) {
        const match = inputValue.match(hourOnlyPattern);
        const hours = parseInt(match[1]);
        totalMinutes = hours * 60;
    } else if (minuteOnlyPattern.test(inputValue)) {
        const match = inputValue.match(minuteOnlyPattern);
        totalMinutes = parseInt(match[1]);
    } else if (!isNaN(inputValue)) {
        // If the input is just a number without "HOUR" or "MINUTE"
        totalMinutes = parseInt(inputValue);
    }

    inputElement.value = totalMinutes;
}
function convertToUpper(inputId) {
    const inputElement = document.getElementById(inputId);
    const inputValue = inputElement.value.trim();
    inputElement.value = inputValue.toUpperCase();
}

function handleSubmit(event) {
    convertToMinutes('HOURS_STUDY');
    convertToMinutes('EXTRA_ACTIVITIES_HOURS');
    convertToUpper('AVG_GRADE_HS');
    convertToUpper('SCHOOL_TYPE');
    convertToUpper('MAJOR');
    convertToUpper('GAP_BEFORE_DEGREE');
    convertToUpper('STUDY_SCHEDULE');
    convertToUpper('PART_TIME_JOB');
    convertToUpper('MOTIVATION');
    convertToUpper('STRESS_MANAGEMENT');
}
document.addEventListener('DOMContentLoaded', function() {
    const container = document.querySelector('body');
    const content = document.querySelector('.content');
    const scrollUpBtn = document.querySelector('.scroll-up');
    const scrollDownBtn = document.querySelector('.scroll-down');

    scrollUpBtn.addEventListener('click', function() {
        container.scrollTop -= 50; // Adjust scrolling speed (50 is an example)
        
    });

    scrollDownBtn.addEventListener('click', function() {
        container.scrollTop += 50; // Adjust scrolling speed (50 is an example)
    });
});
