#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for AI Study Platform
Tests all endpoints using the public URL to ensure user-facing functionality works correctly.
"""

import requests
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional

class StudyPlatformTester:
    def __init__(self, base_url: str = "https://smartstudy-124.preview.emergentagent.com"):
        self.base_url = base_url
        self.token = None
        self.user_id = None
        self.test_user_email = f"test_user_{datetime.now().strftime('%H%M%S')}@example.com"
        self.test_user_password = "TestPass123!"
        self.test_user_name = "Test User"
        
        # Test data storage
        self.material_id = None
        self.flashcard_ids = []
        self.quiz_id = None
        self.note_id = None
        
        # Test results
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []
        self.critical_issues = []
        self.flaky_endpoints = []

    def log_test(self, name: str, success: bool, details: str = "", is_critical: bool = False):
        """Log test results"""
        self.tests_run += 1
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"\n{status} - {name}")
        if details:
            print(f"   Details: {details}")
        
        if success:
            self.tests_passed += 1
        else:
            self.failed_tests.append({"test": name, "details": details, "critical": is_critical})
            if is_critical:
                self.critical_issues.append({"test": name, "details": details})

    def make_request(self, method: str, endpoint: str, data: Any = None, params: Dict = None, expected_status: int = 200) -> tuple[bool, Dict]:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}/api/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method == 'POST':
                if isinstance(data, dict):
                    response = requests.post(url, json=data, headers=headers, params=params, timeout=30)
                else:
                    response = requests.post(url, data=data, headers=headers, params=params, timeout=30)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, params=params, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, params=params, timeout=30)
            else:
                return False, {"error": f"Unsupported method: {method}"}
            
            success = response.status_code == expected_status
            try:
                response_data = response.json() if response.content else {}
            except:
                response_data = {"raw_response": response.text}
            
            if not success:
                response_data["status_code"] = response.status_code
                response_data["expected_status"] = expected_status
            
            return success, response_data
            
        except requests.exceptions.Timeout:
            return False, {"error": "Request timeout (30s)"}
        except requests.exceptions.ConnectionError:
            return False, {"error": "Connection error - server may be down"}
        except Exception as e:
            return False, {"error": f"Request failed: {str(e)}"}

    def test_user_registration(self):
        """Test user registration"""
        success, response = self.make_request(
            'POST', 
            'auth/register',
            {
                "email": self.test_user_email,
                "password": self.test_user_password,
                "name": self.test_user_name
            },
            expected_status=200
        )
        
        if success and 'token' in response:
            self.token = response['token']
            self.user_id = response['user']['id']
            self.log_test("User Registration", True, f"User created with ID: {self.user_id}")
        else:
            self.log_test("User Registration", False, f"Response: {response}", is_critical=True)
        
        return success

    def test_user_login(self):
        """Test user login"""
        success, response = self.make_request(
            'POST',
            'auth/login',
            {
                "email": self.test_user_email,
                "password": self.test_user_password
            },
            expected_status=200
        )
        
        if success and 'token' in response:
            self.token = response['token']
            self.log_test("User Login", True, "Login successful")
        else:
            self.log_test("User Login", False, f"Response: {response}", is_critical=True)
        
        return success

    def test_create_material(self):
        """Test creating study material"""
        material_data = {
            "title": "Test Study Material",
            "content": "This is a comprehensive test material about artificial intelligence. AI is a branch of computer science that aims to create intelligent machines. Machine learning is a subset of AI that enables computers to learn without being explicitly programmed. Deep learning uses neural networks with multiple layers to model and understand complex patterns."
        }
        
        success, response = self.make_request(
            'POST',
            'materials',
            params=material_data,
            expected_status=200
        )
        
        if success and 'id' in response:
            self.material_id = response['id']
            self.log_test("Create Material", True, f"Material created with ID: {self.material_id}")
        else:
            self.log_test("Create Material", False, f"Response: {response}", is_critical=True)
        
        return success

    def test_get_materials(self):
        """Test retrieving materials"""
        success, response = self.make_request('GET', 'materials')
        
        if success and isinstance(response, list):
            self.log_test("Get Materials", True, f"Retrieved {len(response)} materials")
        else:
            self.log_test("Get Materials", False, f"Response: {response}")
        
        return success

    def test_generate_flashcards(self):
        """Test AI flashcard generation"""
        if not self.material_id:
            self.log_test("Generate Flashcards", False, "No material ID available", is_critical=True)
            return False
        
        success, response = self.make_request(
            'POST',
            'flashcards/generate',
            params={"material_id": self.material_id},
            expected_status=200
        )
        
        if success and isinstance(response, list) and len(response) > 0:
            self.flashcard_ids = [card['id'] for card in response]
            self.log_test("Generate Flashcards", True, f"Generated {len(response)} flashcards")
        else:
            self.log_test("Generate Flashcards", False, f"Response: {response}", is_critical=True)
        
        return success

    def test_get_flashcards(self):
        """Test retrieving flashcards"""
        success, response = self.make_request('GET', 'flashcards')
        
        if success and isinstance(response, list):
            self.log_test("Get Flashcards", True, f"Retrieved {len(response)} flashcards")
        else:
            self.log_test("Get Flashcards", False, f"Response: {response}")
        
        return success

    def test_review_flashcard(self):
        """Test flashcard review (spaced repetition)"""
        if not self.flashcard_ids:
            self.log_test("Review Flashcard", False, "No flashcard IDs available")
            return False
        
        flashcard_id = self.flashcard_ids[0]
        success, response = self.make_request(
            'POST',
            f'flashcards/{flashcard_id}/review',
            params={"quality": 4},
            expected_status=200
        )
        
        if success and 'message' in response:
            self.log_test("Review Flashcard", True, "Flashcard review recorded")
        else:
            self.log_test("Review Flashcard", False, f"Response: {response}")
        
        return success

    def test_generate_quiz(self):
        """Test AI quiz generation"""
        if not self.material_id:
            self.log_test("Generate Quiz", False, "No material ID available", is_critical=True)
            return False
        
        success, response = self.make_request(
            'POST',
            'quizzes/generate',
            params={"material_id": self.material_id},
            expected_status=200
        )
        
        if success and 'id' in response and 'questions' in response:
            self.quiz_id = response['id']
            self.log_test("Generate Quiz", True, f"Generated quiz with {len(response['questions'])} questions")
        else:
            self.log_test("Generate Quiz", False, f"Response: {response}", is_critical=True)
        
        return success

    def test_get_quizzes(self):
        """Test retrieving quizzes"""
        success, response = self.make_request('GET', 'quizzes')
        
        if success and isinstance(response, list):
            self.log_test("Get Quizzes", True, f"Retrieved {len(response)} quizzes")
        else:
            self.log_test("Get Quizzes", False, f"Response: {response}")
        
        return success

    def test_submit_quiz(self):
        """Test quiz submission"""
        if not self.quiz_id:
            self.log_test("Submit Quiz", False, "No quiz ID available")
            return False
        
        # Submit answers (assuming 5 questions, answering all with option 0)
        answers = [0, 1, 0, 1, 0]
        success, response = self.make_request(
            'POST',
            f'quizzes/{self.quiz_id}/submit',
            answers,
            expected_status=200
        )
        
        if success and 'score' in response:
            self.log_test("Submit Quiz", True, f"Quiz submitted, score: {response['score']}%")
        else:
            self.log_test("Submit Quiz", False, f"Response: {response}")
        
        return success

    def test_create_note(self):
        """Test creating a note"""
        note_data = {
            "title": "Test Note",
            "content": "This is a test note for the AI study platform."
        }
        
        success, response = self.make_request(
            'POST',
            'notes',
            params=note_data,
            expected_status=200
        )
        
        if success and 'id' in response:
            self.note_id = response['id']
            self.log_test("Create Note", True, f"Note created with ID: {self.note_id}")
        else:
            self.log_test("Create Note", False, f"Response: {response}")
        
        return success

    def test_summarize_material(self):
        """Test AI summarization"""
        if not self.material_id:
            self.log_test("Summarize Material", False, "No material ID available", is_critical=True)
            return False
        
        success, response = self.make_request(
            'POST',
            'notes/summarize',
            params={"material_id": self.material_id},
            expected_status=200
        )
        
        if success and 'summary' in response:
            self.log_test("Summarize Material", True, "AI summary generated")
        else:
            self.log_test("Summarize Material", False, f"Response: {response}", is_critical=True)
        
        return success

    def test_get_notes(self):
        """Test retrieving notes"""
        success, response = self.make_request('GET', 'notes')
        
        if success and isinstance(response, list):
            self.log_test("Get Notes", True, f"Retrieved {len(response)} notes")
        else:
            self.log_test("Get Notes", False, f"Response: {response}")
        
        return success

    def test_chat_with_ai(self):
        """Test AI tutor chat"""
        success, response = self.make_request(
            'POST',
            'chat',
            params={"message": "What is artificial intelligence?"},
            expected_status=200
        )
        
        if success and 'message' in response:
            self.log_test("AI Chat", True, "AI tutor responded successfully")
        else:
            self.log_test("AI Chat", False, f"Response: {response}", is_critical=True)
        
        return success

    def test_get_chat_history(self):
        """Test retrieving chat history"""
        success, response = self.make_request('GET', 'chat/history')
        
        if success and isinstance(response, list):
            self.log_test("Get Chat History", True, f"Retrieved {len(response)} messages")
        else:
            self.log_test("Get Chat History", False, f"Response: {response}")
        
        return success

    def test_create_study_session(self):
        """Test creating study session"""
        success, response = self.make_request(
            'POST',
            'sessions',
            params={"duration": 30, "focus_mode": True},
            expected_status=200
        )
        
        if success and 'id' in response:
            self.log_test("Create Study Session", True, "Study session created")
        else:
            self.log_test("Create Study Session", False, f"Response: {response}")
        
        return success

    def test_get_sessions(self):
        """Test retrieving study sessions"""
        success, response = self.make_request('GET', 'sessions')
        
        if success and isinstance(response, list):
            self.log_test("Get Sessions", True, f"Retrieved {len(response)} sessions")
        else:
            self.log_test("Get Sessions", False, f"Response: {response}")
        
        return success

    def test_get_progress(self):
        """Test retrieving user progress"""
        success, response = self.make_request('GET', 'progress')
        
        if success and 'user_id' in response:
            self.log_test("Get Progress", True, "Progress data retrieved")
        else:
            self.log_test("Get Progress", False, f"Response: {response}")
        
        return success

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚀 Starting AI Study Platform Backend Tests")
        print(f"📍 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Authentication Tests
        print("\n📋 AUTHENTICATION TESTS")
        if not self.test_user_registration():
            print("❌ Registration failed - stopping tests")
            return self.generate_report()
        
        if not self.test_user_login():
            print("❌ Login failed - stopping tests")
            return self.generate_report()
        
        # Core Feature Tests
        print("\n📚 MATERIALS TESTS")
        self.test_create_material()
        self.test_get_materials()
        
        print("\n🃏 FLASHCARDS TESTS")
        self.test_generate_flashcards()
        self.test_get_flashcards()
        self.test_review_flashcard()
        
        print("\n📝 QUIZ TESTS")
        self.test_generate_quiz()
        self.test_get_quizzes()
        self.test_submit_quiz()
        
        print("\n📄 NOTES TESTS")
        self.test_create_note()
        self.test_summarize_material()
        self.test_get_notes()
        
        print("\n🤖 AI TUTOR TESTS")
        self.test_chat_with_ai()
        self.test_get_chat_history()
        
        print("\n⏱️ STUDY SESSION TESTS")
        self.test_create_study_session()
        self.test_get_sessions()
        
        print("\n📊 PROGRESS TESTS")
        self.test_get_progress()
        
        return self.generate_report()

    def generate_report(self):
        """Generate test report"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        print(f"✅ Tests Passed: {self.tests_passed}/{self.tests_run} ({success_rate:.1f}%)")
        
        if self.critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(self.critical_issues)}):")
            for issue in self.critical_issues:
                print(f"   • {issue['test']}: {issue['details']}")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS ({len(self.failed_tests)}):")
            for test in self.failed_tests:
                criticality = " [CRITICAL]" if test['critical'] else ""
                print(f"   • {test['test']}{criticality}: {test['details']}")
        
        # Determine overall status
        if len(self.critical_issues) > 0:
            print(f"\n🔴 OVERALL STATUS: CRITICAL ISSUES FOUND")
            return 1
        elif success_rate < 70:
            print(f"\n🟡 OVERALL STATUS: MULTIPLE FAILURES")
            return 1
        elif success_rate < 100:
            print(f"\n🟡 OVERALL STATUS: SOME ISSUES")
            return 0
        else:
            print(f"\n🟢 OVERALL STATUS: ALL TESTS PASSED")
            return 0

def main():
    """Main test execution"""
    tester = StudyPlatformTester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())