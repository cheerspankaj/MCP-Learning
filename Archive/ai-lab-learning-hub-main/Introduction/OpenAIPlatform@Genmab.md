## **Accelerate, Enable, and Transform Work with the OpenAI Platform and API**

Welcome to this session on OpenAI's platform and API. Today, we’ll explore how OpenAI’s offerings can empower you to create innovative solutions, streamline workflows, and enhance user experiences. Projects like Draft0 and AITranslator, showcase its potential to transform how we work.

### **OpenAI's Two Key Products**

1. **ChatGPT**
   ChatGPT is OpenAI's conversational AI assistant. It offers a suite of powerful features designed to enhance productivity and collaboration:

   - **Canvas:** A tool for collaborative ideation and structured problem-solving. *(Consider adding a screenshot of the Canvas interface here.)*
   - **File Uploads:** Enables interaction with uploaded data directly. *(Consider including a visual of the file upload interface here.)*
   - **Conversation History:** Allows you to maintain and revisit contextualized discussions.
   - **CustomGPT:** Lets you tailor the AI to specific tasks or organizational needs. *(A screenshot of the customization options might be helpful here.)*
   - **Code Interpreter:** A versatile tool for data analysis, calculations, and more.

   These features make ChatGPT ideal for interactive problem-solving and productivity enhancement.

2. **OpenAI Platform**
   The platform provides direct access to OpenAI models, giving you greater control over interactions. This makes it ideal for building custom user experiences and embedding AI capabilities directly into applications.\
   **Key Difference:** While ChatGPT is a ready-to-use assistant, the platform enables you to work directly with models, tailoring functionality to meet specific requirements.

---

### **Accessing the Genmab OpenAI Platform**

Platform access is not granted by default. Here’s how the process works at Genmab:

1. **Submit a Consultation Request**

   - Provide details about your use case via a consultation request. 
   - Clearly define your objectives to help us understand how the API will support your work.

2. **Intake Meeting**

   - We’ll schedule a session to discuss your use case in detail.
   - This ensures alignment on expectations and project outcomes.

3. **Access to Genmab OpenAI API Key**

   - Once approved, we initiate the process to secure your API key.

---

### **Working Within Genmab's Framework**

At Genmab, we use **projects** to organize and manage work related to the OpenAI platform. Here’s how you’ll get started:

1. **Project Assignment**

   - A new project will be created for your use case, or an existing project aligned with your department will be identified.
   - This will serve as your **home project**, where all work related to the OpenAI API must be conducted.

2. **Platform Invitation**

   - You’ll receive an email invitation to join the platform.
   - Accept the invite and log in using **SSO (Single Sign-On)**.

3. **Project Integration**

   - After logging in, you will shortly be added to the appropriate project for your use case.
   - You’ll also be added to the **private GenAI App Builder channel** within the ChatGPT Teams group, where you can collaborate with peers and share insights.

4. **Verify Project Assignment**

   - Confirm that you are in your home project by checking the top right of the interface. It should display: **Genmab >> [Your Home Project]**. If not, switch to your home project by clicking the displayed project name and selecting the correct project from the dropdown menu. *(Adding a screenshot of the project assignment view could clarify this step.)*

5. **Avoid Default Project Usage**

   - Do not create any assets in the default project as these will be deleted. Always switch to your home project before proceeding.

6. **Create Your API Key**

   - Navigate to the "API Keys" page in the left navigation bar and create your first key. *(Consider including a screenshot of the API Keys page to guide users visually.)*

   For any API keys that you create, follow these guidelines:

   - **Switch to Your Home Project:** Ensure the key is created under your assigned home project. Keys created in the default project will be removed.
   - **Naming Convention:**
     - Prefix the key name with your ID. For example: `sany_dev` (Sanyam's key for development).
     - Add the project name (if applicable) after your ID. For example: `guma-PP-TQSDEV` (owner: Guma, app: Power Platform, dept: TQS, env: DEV).

7. **Architectural Standards for API Keys**

   - **No Sharing:** API keys should be personal to ensure proper tracking and cost attribution.
   - **Use for Local Development:** Developers should use personal keys for local work.
   - **Avoid Storing Keys in Git:** Use a `.env` file for key storage and ensure `.gitignore` includes it.
   - **Service Keys for Deployed Applications:** For deployed apps, use service-account API keys. Post a request in the appropriate channel to obtain one.
   - **AWS Secrets Manager:** Securely store and access keys for deployed apps via AWS Secrets Manager.

   These practices help maintain security, cost accountability, and operational efficiency.

---

### **Additional Resources**

- **FAQs:** Consult the [New API Users FAQ](#) for common issues and troubleshooting.

- **AI Lab Learning Hub:** Complete the OpenAI course available in the Learning Hub.

- **Monitor Usage:** Use the "Usage" tab in the left navigation bar to track API usage and costs. Refer to OpenAI’s [Pricing Page](https://openai.com/api/pricing/) for detailed pricing information.

  **Pro Tip:** Choose the most cost-effective model for your needs. For example, `gpt-4-turbo` is significantly cheaper than `gpt-4` for most tasks.

- **Monthly API Users Meeting:** Attend these sessions to share progress, insights, and best practices.



**Pro Tip**

ChatGPT is an all-you-can-use tool. We pay a fixed price for unlimited usage.

The API, however, operates on a pay-per-use model. To ensure cost-efficiency:

- Use ChatGPT for learning how to interact with large language models (LLMs) or for refining your prompts.
- Refine prompts in ChatGPT before moving to the API. Once you have an optimized prompt, you can transition it to the API.

This approach helps avoid unnecessary API costs associated with learning and experimentation in prompt engineering.

---

### **Takeaways**

This process ensures that your work with the OpenAI platform is streamlined, secure, and aligned with Genmab’s operational goals. By following this structured approach, you can leverage OpenAI’s transformative capabilities effectively.

Let’s unlock the power of OpenAI together!

