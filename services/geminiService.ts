import { GoogleGenAI, Type } from "@google/genai";
import { ParsedResume, LearningResource, QuizQuestion, JobListing } from "../types";

const apiKey = process.env.API_KEY || '';
const ai = new GoogleGenAI({ apiKey });

const MODEL_FLASH = 'gemini-2.5-flash';

export const parseResumeDocument = async (base64Data: string, mimeType: string): Promise<ParsedResume> => {
  const prompt = `
    Analyze this document. It could be a Resume, a CV, or an Employment Offer Letter.
    
    1. If it is a **Resume/CV**: Extract the candidate's details normally.
    2. If it is an **Offer Letter** or **Job Description**: 
       - Extract the candidate name (if present).
       - Treat the "Job Role" mentioned as a "Suggested Role".
       - Treat the "Required Skills" mentioned in the letter as the candidate's current or required skills.
       - Construct a summary based on the role offered.
    
    Return structured JSON with:
    1. Full Name
    2. A professional summary (max 50 words)
    3. Key skills (array of strings) - if an offer letter, list skills mentioned in the letter (e.g. ML, Deep Learning, Data Analysis).
    4. Total years of experience (number) - estimate or set to 0 if intern.
    5. Suggest 3 distinct job roles. If it's an offer letter, the FIRST role MUST be the one offered in the document.
       For each role, provide a match score (0-100), a brief reasoning, and a list of 5 required technical skills.
  `;

  const response = await ai.models.generateContent({
    model: MODEL_FLASH,
    contents: {
      parts: [
        { inlineData: { mimeType, data: base64Data } },
        { text: prompt }
      ]
    },
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          fullName: { type: Type.STRING },
          summary: { type: Type.STRING },
          skills: { type: Type.ARRAY, items: { type: Type.STRING } },
          yearsOfExperience: { type: Type.NUMBER },
          suggestedRoles: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                title: { type: Type.STRING },
                matchScore: { type: Type.NUMBER },
                reasoning: { type: Type.STRING },
                requiredSkills: { type: Type.ARRAY, items: { type: Type.STRING } }
              }
            }
          }
        }
      }
    }
  });

  return JSON.parse(response.text || '{}') as ParsedResume;
};

export const generateLearningPath = async (role: string, currentSkills: string[]): Promise<LearningResource[]> => {
  const prompt = `
    Create a study plan for a candidate targeting the role of "${role}".
    Their current skills are: ${currentSkills.join(', ')}.
    Provide a list of 5 high-quality learning resources (Courses, Articles, Documentation) to close skill gaps.
  `;

  const response = await ai.models.generateContent({
    model: MODEL_FLASH,
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING },
            type: { type: Type.STRING, enum: ['Course', 'Article', 'Video', 'Documentation'] },
            url: { type: Type.STRING },
            description: { type: Type.STRING }
          }
        }
      }
    }
  });

  return JSON.parse(response.text || '[]') as LearningResource[];
};

export const generateQuiz = async (role: string, difficulty: 'Easy' | 'Medium' | 'Hard', topic: string = 'Technical'): Promise<QuizQuestion[]> => {
  const prompt = `
    Generate a ${difficulty} ${topic} quiz for a "${role}" interview.
    Create 5 multiple-choice questions.
  `;

  const response = await ai.models.generateContent({
    model: MODEL_FLASH,
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            question: { type: Type.STRING },
            options: { type: Type.ARRAY, items: { type: Type.STRING } },
            correctAnswer: { type: Type.INTEGER },
            explanation: { type: Type.STRING },
            category: { type: Type.STRING, enum: ['Technical', 'Aptitude', 'Behavioral'] }
          }
        }
      }
    }
  });

  return JSON.parse(response.text || '[]') as QuizQuestion[];
};

export const generateAptitudePrep = async (): Promise<LearningResource[]> => {
  const prompt = `
    Provide 5 general aptitude and logical reasoning preparation topics and resources for a job interview.
    Focus on quantitative aptitude, logical reasoning, and verbal ability.
  `;

  const response = await ai.models.generateContent({
    model: MODEL_FLASH,
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING },
            type: { type: Type.STRING, enum: ['Article', 'Video', 'Documentation'] },
            url: { type: Type.STRING },
            description: { type: Type.STRING }
          }
        }
      }
    }
  });

  return JSON.parse(response.text || '[]') as LearningResource[];
};

export const generateFullMockTest = async (role: string): Promise<QuizQuestion[]> => {
  const prompt = `
    Create a comprehensive 10-question mock test for a "${role}" candidate.
    Include:
    - 5 Technical questions related to the role
    - 3 Aptitude/Logic questions
    - 2 Behavioral/Situational questions
    Return as a JSON array.
  `;

  const response = await ai.models.generateContent({
    model: MODEL_FLASH,
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            question: { type: Type.STRING },
            options: { type: Type.ARRAY, items: { type: Type.STRING } },
            correctAnswer: { type: Type.INTEGER },
            explanation: { type: Type.STRING },
            category: { type: Type.STRING, enum: ['Technical', 'Aptitude', 'Behavioral'] }
          }
        }
      }
    }
  });

  return JSON.parse(response.text || '[]') as QuizQuestion[];
};

export const searchJobs = async (role: string, location: string = "Remote"): Promise<JobListing[]> => {
  const prompt = `Find 5 recent job listings for "${role}" in "${location}".`;

  const response = await ai.models.generateContent({
    model: MODEL_FLASH,
    contents: prompt,
    config: {
      tools: [{ googleSearch: {} }]
    }
  });

  const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
  
  const listings: JobListing[] = chunks
    .filter(c => c.web?.uri && c.web?.title)
    .map(c => ({
      title: c.web?.title || "Job Opening",
      company: "Source: Web",
      location: location,
      url: c.web?.uri || "#"
    }))
    .slice(0, 6);
  
  // Fallback if no chunks found (rare but possible if model just chats)
  if (listings.length === 0) {
      return [
          { title: `${role} - Search Results`, company: "Google Search", location, url: `https://www.google.com/search?q=${encodeURIComponent(role + ' jobs')}` }
      ];
  }

  return listings;
};