// Course data for Pre-PLP Academy
const courses = [
    {
        id: 101,
        title: "Foundations of Programming",
        description: "Learn the basics of programming logic and structure.",
        lessons: [
            { id: 1, title: "Introduction to Algorithms" },
            { id: 2, title: "Data Types and Variables" },
            { id: 3, title: "Control Structures" }
        ]
    },
    {
        id: 102,
        title: "Web Development Essentials",
        description: "Build modern websites with HTML, CSS, and JavaScript.",
        lessons: [
            { id: 1, title: "HTML5 Fundamentals" },
            { id: 2, title: "CSS3 Styling" },
            { id: 3, title: "JavaScript Basics" }
        ]
    }
];

// Track completed lessons
let userProgress = JSON.parse(localStorage.getItem('preplpProgress')) || [];

// Render all courses on the home page
function renderCourses() {
    const container = document.getElementById('courses-container');
    container.innerHTML = courses.map(course => `
        <div class="course-card" onclick="openCourse(${course.id})">
            <h3>${course.title}</h3>
            <p>${course.description}</p>
            <button>Start Course</button>
        </div>
    `).join('');
}

// Open a course and show its lessons
function openCourse(courseId) {
    const course = courses.find(c => c.id === courseId);
    const detailTitle = document.getElementById('course-title-detail');
    const lessonsContainer = document.getElementById('lessons-container');

    detailTitle.textContent = course.title;
    lessonsContainer.innerHTML = course.lessons.map(lesson => {
        const isCompleted = userProgress.some(p => p.courseId === courseId && p.lessonId === lesson.id);
        return `
            <div class="lesson-card ${isCompleted ? 'completed' : ''}">
                <h4>${lesson.title}</h4>
                <button onclick="toggleLesson(${courseId}, ${lesson.id}, event)">
                    ${isCompleted ? '✓ Completed' : 'Mark as Completed'}
                </button>
            </div>
        `;
    }).join('');

    document.getElementById('courses-section').classList.add('hidden');
    document.getElementById('course-detail-section').classList.remove('hidden');
}

// Toggle lesson completion status
function toggleLesson(courseId, lessonId, event) {
    event.stopPropagation();
    const index = userProgress.findIndex(p => p.courseId === courseId && p.lessonId === lessonId);
    if (index === -1) {
        userProgress.push({ courseId, lessonId });
    } else {
        userProgress.splice(index, 1);
    }
    localStorage.setItem('preplpProgress', JSON.stringify(userProgress));
    openCourse(courseId); // Refresh the view
}

// Go back to the courses list
document.getElementById('back-to-courses').addEventListener('click', () => {
    document.getElementById('course-detail-section').classList.add('hidden');
    document.getElementById('courses-section').classList.remove('hidden');
});

// Initialize the app
renderCourses();
