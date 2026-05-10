using ai.Models;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;

namespace ai.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
        }

        

        public IActionResult Index()
        {
            var dummyNews = new List<NewsArticle>
    {
        new() { Title = "GPT-5 Release Date Leaked", Summary = "OpenAI hints at multimodal breakthrough...", PublishDate = DateTime.Now, Url = "#" },
        new() { Title = "AI Video Editing Wars", Summary = "Runway vs Pika – which one wins?", PublishDate = DateTime.Now.AddDays(-1), Url = "#" },
        new() { Title = "LLaMA 3.2 Explained", Summary = "Meta's new on-device models.", PublishDate = DateTime.Now.AddDays(-2), Url = "#" },
        new() { Title = "AI in Healthcare Diagnostics", Summary = "New study shows 96% accuracy.", PublishDate = DateTime.Now.AddDays(-3), Url = "#" },
        // Add 6-8 more to see the grid fill nicely
    };
            return View(dummyNews);
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
